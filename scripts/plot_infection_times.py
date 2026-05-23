import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_eicu_infection_times(eicu_raw_dir, cohort_path):
    print("Loading eICU data...")
    cohort = pd.read_csv(cohort_path)
    diag = pd.read_csv(Path(eicu_raw_dir) / "diagnosis.csv.gz", usecols=["patientunitstayid", "icd9code", "diagnosisoffset"], low_memory=False)
    
    # Filter to cohort
    diag = diag[diag["patientunitstayid"].isin(cohort["patientunitstayid"])]
    
    # Very basic string match for ICD-9 prefixes
    sepsis_mask = diag["icd9code"].astype(str).str.contains(r"^(038|995\.9)", regex=True, na=False)
    uti_mask = diag["icd9code"].astype(str).str.contains(r"^(599\.0)", regex=True, na=False)
    
    sepsis_times = diag[sepsis_mask]["diagnosisoffset"] / (60 * 24) # convert minutes to days
    uti_times = diag[uti_mask]["diagnosisoffset"] / (60 * 24)
    
    # Remove negative times (prevalent)
    sepsis_times = sepsis_times[sepsis_times > 0].dropna()
    uti_times = uti_times[uti_times > 0].dropna()
    
    return sepsis_times, uti_times


def get_mimiciv_infection_times(mimic_raw_dir, cohort_path):
    print("Loading MIMIC-IV data...")
    cohort = pd.read_csv(cohort_path, parse_dates=["intime"])
    micro = pd.read_csv(Path(mimic_raw_dir) / "hosp" / "microbiologyevents.csv.gz", 
                        usecols=["hadm_id", "charttime", "spec_type_desc"], 
                        parse_dates=["charttime"], low_memory=False)
    
    micro = micro[micro["hadm_id"].isin(cohort["hadm_id"])].dropna(subset=["charttime", "hadm_id"])
    
    # Merge intime
    micro = micro.merge(cohort[["hadm_id", "intime"]], on="hadm_id", how="inner")
    micro["days"] = (micro["charttime"] - micro["intime"]).dt.total_seconds() / 86400.0
    
    # Filter for incident only
    micro = micro[micro["days"] > 0]
    
    sepsis_mask = micro["spec_type_desc"].astype(str).str.contains("blood", case=False, na=False)
    uti_mask = micro["spec_type_desc"].astype(str).str.contains("urine", case=False, na=False)
    
    sepsis_times = micro[sepsis_mask]["days"]
    uti_times = micro[uti_mask]["days"]
    
    return sepsis_times, uti_times

def main():
    eicu_raw = "/home/mpham/workspace/source/ehrfm/survpfn/ehr-meds/data/raw/eicu/2.0"
    eicu_cohort = "/home/mpham/workspace/source/ehrfm/survpfn/ehr-meds/data/processed_hcai/eicu/hcai_cohort_eicu.csv"
    
    mimic_raw = "/home/mpham/workspace/source/ehrfm/survpfn/ehr-meds/data/raw/mimiciv/3.1"
    mimic_cohort = "/home/mpham/workspace/source/ehrfm/survpfn/ehr-meds/data/processed_hcai/mimiciv/hcai_cohort_mimiciv.csv"
    
    eicu_sep, eicu_uti = get_eicu_infection_times(eicu_raw, eicu_cohort)
    mimic_sep, mimic_uti = get_mimiciv_infection_times(mimic_raw, mimic_cohort)
    
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    def plot_hist(ax, data, title, color):
        ax.hist(data[data < 30], bins=30, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(f"{title}\nAvg: {data.mean():.1f} days, Median: {data.median():.1f} days")
        ax.set_xlabel("Days since ICU Admission")
        ax.set_ylabel("Frequency")
        ax.axvline(data.median(), color='red', linestyle='dashed', linewidth=2, label=f'Median {data.median():.1f}d')
        ax.legend()

    plot_hist(axs[0, 0], eicu_sep, "eICU Sepsis (Diagnosis Offset)", 'salmon')
    plot_hist(axs[0, 1], eicu_uti, "eICU UTI (Diagnosis Offset)", 'skyblue')
    
    plot_hist(axs[1, 0], mimic_sep, "MIMIC-IV Blood Cultures (Proxy for Sepsis)", 'salmon')
    plot_hist(axs[1, 1], mimic_uti, "MIMIC-IV Urine Cultures (Proxy for UTI)", 'skyblue')
    
    plt.tight_layout()
    out_img = "/home/mpham/workspace/source/ehrfm/ehrpfn-inference/infection_times.png"
    plt.savefig(out_img, dpi=300)
    print(f"\nPlot successfully saved to: {out_img}")
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"eICU SEPSIS:   Median {eicu_sep.median():.1f} days, Avg {eicu_sep.mean():.1f} days")
    print(f"eICU UTI:      Median {eicu_uti.median():.1f} days, Avg {eicu_uti.mean():.1f} days")
    print(f"MIMIC SEPSIS:  Median {mimic_sep.median():.1f} days, Avg {mimic_sep.mean():.1f} days")
    print(f"MIMIC UTI:     Median {mimic_uti.median():.1f} days, Avg {mimic_uti.mean():.1f} days")

if __name__ == "__main__":
    main()
