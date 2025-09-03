from typing import *
import polars as pl
from pathlib import Path
import os
import os.path as osp
import json

from ..patterns import ScanAndAggregate
from ..utils import static_class

from loguru import logger
from sklearn.model_selection import train_test_split


@static_class
class LabelGenerator(ScanAndAggregate):
    needed_cols:List[str] = [
        "subject_id",
        "admission_id",
        "time",
        "code",
    ]
    def __call__(
            self, df: pl.DataFrame, tasks:List[str], 
            do_remove_no_hcai_cases_in_early_admissions:bool=False,
            neg_over_pos_thresh:int = 10,
            does_split:bool=False, ratio:float=1.0, seed:int=42) -> pl.DataFrame:
        self.tasks = tasks
        self.generators = []
        for task in self.tasks:
            if task.startswith("next_admission_icd"):
                cohort_type = task.split(":")[-1]
                self.generators.append(NextAdmissionICDCodeBinaryClassificationLabelCreator(
                    cohort_type, does_split=does_split, ratio=ratio, seed=seed, neg_over_pos_thresh=neg_over_pos_thresh,
                    do_remove_no_hcai_cases_in_early_admissions=do_remove_no_hcai_cases_in_early_admissions
                ))
        return df.select(self.needed_cols)
    
    def agg(self, in_fps: list, out_fp: str | Path) -> None:
        if getattr(self, "generators", None) is None:
            logger.warning("No generators found. Please call the generator first.")
            return
        os.makedirs(out_fp, exist_ok=True)
        df = pl.scan_parquet(in_fps).collect()
        for generator in self.generators:
            generator(df, output_folder=out_fp)


class NextAdmissionICDCodeBinaryClassificationLabelCreator:
    def __init__(
            self, cohort_type:str, 
            does_split:bool=False, 
            do_remove_no_hcai_cases_in_early_admissions:bool = False,
            neg_over_pos_thresh:int=10,
            ratio:float=1.0, seed:int=42
        ):
        self.cohort_type = cohort_type
        self.does_split = does_split
        self.do_remove_no_hcai_cases_in_early_admissions = do_remove_no_hcai_cases_in_early_admissions
        self.ratio = ratio
        self.neg_over_pos_thresh = neg_over_pos_thresh
        self.seed = seed
    
    def __call__(self, df: pl.DataFrame, output_folder: str | Path) -> None:
        output_folder = osp.join(output_folder, self.cohort_type)
        os.makedirs(output_folder, exist_ok=True)
        result = self.process_meds_data(
            df,
            output_folder=output_folder,
            cohort_type=self.cohort_type
        )
    

    def create_labels_meds(self, df: pl.DataFrame, target_codes: List[str]) -> pl.DataFrame:
        """
        Create prediction labels for the next admission of the same subject_id.

        Args:
            df (pl.DataFrame): The input DataFrame containing medications data.
            target_codes (List[str]): The list of codes to check for in the next admission.

        Returns:
            pl.DataFrame: A DataFrame with admission_id and binary prediction labels.
        """
        # Ensure the DataFrame is sorted by subject_id, admission_id, and time
        df = df.sort(["subject_id", "admission_id", "time"])

        # Group by subject_id and admission_id, and collect all codes for each admission
        grouped = df.group_by(["subject_id", "admission_id"]).agg(
            pl.col("code").alias("codes")
        )
        # Shift the codes column to get the next admission's codes for each admission
        # grouped = grouped.with_columns(
        #     pl.col("codes").shift(-1).over("subject_id").alias("next_admission_codes")
        # )


        # grouped = (
        #     grouped.with_columns(
        #     code_of_interests_next_admission=(
        #         pl.col("next_admission_codes")
        #         .list.set_intersection(target_codes)
        #     ))
        #     .with_columns(
        #         (pl.col("code_of_interests_next_admission").list.len()>0).cast(pl.Int32).alias(f"prediction_label")
        #     )
        # )

        grouped = (
            grouped.with_columns(
            code_of_interests_this_admission=(
                pl.col("codes")
                .list.set_intersection(target_codes)
            ))
            .with_columns(
                (pl.col("code_of_interests_this_admission").list.len()>0).cast(pl.Boolean).alias(f"prediction_label")
            )
        )

        # Select only admission_id and label for the output
        result = grouped.select(["subject_id", "admission_id", "prediction_label"])
        result = result.drop_nulls("prediction_label")

        return result

    def process_meds_data(self, df: pl.DataFrame, output_folder: str, cohort_type:str) -> None:
        """
        Process MEDS data and create labels for the next admission.

        Args:
            input_folder (str): Path to the input folder containing MEDS data.
            output_folder (str): Path to the output folder for results.
        """

        assert cohort_type in DISEASE_GROUP_TO_ICDS, f"Invalid cohort_type: {cohort_type}. Valid options are: {list(DISEASE_GROUP_TO_ICDS.keys())}"
        # Define target codes
        target_codes = []
        icd_top_codes = DISEASE_GROUP_TO_ICDS[cohort_type]
        # Convert ICD codes to the format "ICD//CM//{code}"
        for code in icd_top_codes:
            all_codes = list(get_cohort_data_from_json(code).keys())
            all_codes = [f"ICD//CM//10//{code}" for code in all_codes]
            target_codes.extend(all_codes)
            target_codes = list(set(target_codes))

        # Create labels
        results = self.create_labels_meds(
            df=df,
            target_codes=target_codes
        )

        # Class count
        class_count = {}
        label_column = 'prediction_label'
        class_count[label_column] = results.select(label_column).group_by(label_column).agg(pl.count()).sort(label_column)
        count_dict = class_count[label_column].to_dicts()
        class_count[label_column] = {
            str(i[label_column]): int(i["count"])
            for i in count_dict
        }

        # Save the results
        label_infos = {}
        label_infos["target_codes"] = target_codes
        label_infos["class_count"] = class_count

        # Split
        if self.does_split is True or self.does_split == "train":
            train_df, val_df = train_test_split(
                results.to_pandas(),
                train_size=self.ratio,
                random_state=self.seed,
                stratify=results["prediction_label"].to_list()
            )
            train_df = pl.from_pandas(train_df)

            if self.do_remove_no_hcai_cases_in_early_admissions:
                train_df = remove_no_hcai_cases_in_early_admissions(train_df, ratio_threshold=self.neg_over_pos_thresh)


            train_df.write_csv(osp.join(output_folder, "train_prediction_labels.csv"))
            val_df = pl.from_pandas(val_df)
            val_df.write_csv(osp.join(output_folder, "val_prediction_labels.csv"))

            class_count = {}
            tmp = train_df.select(label_column).group_by(label_column).agg(pl.count()).sort(label_column)
            count_dict = tmp.to_dicts()
            class_count['train'] = {
                str(i[label_column]): int(i["count"])
                for i in count_dict
            }
            tmp = val_df.select(label_column).group_by(label_column).agg(pl.count()).sort(label_column)
            count_dict = tmp.to_dicts()
            class_count['val'] = {
                str(i[label_column]): int(i["count"])
                for i in count_dict
            }

            label_infos["split"] = class_count
            
        else:
            # Save the result to a CSV file
            results.write_csv(osp.join(output_folder, "prediction_labels.csv"))

        # Save the result to a CSV file
        with open(osp.join(output_folder, "label_infos.json"), "w") as f:
            json.dump(label_infos, f, indent=4)

        return results


