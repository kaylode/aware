import hydra
import polars as pl
from pathlib import Path
import pickle
from ehrpfn.utils.vocabulary import Vocabulary
from scripts.tokenization.constants import STATIC_DATA_FN, SpecialToken as ST


@hydra.main(version_base=None, config_path="./configs", config_name="mimiciv")
def run(cfg):

    output_dir = Path(cfg.output_dir) / cfg.out_fn
    # Update the real output directory path
    cfg.output_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary()
    list_of_words = []

    if cfg.manually_added_tokens:
        vocab.add_words(cfg.manually_added_tokens)

    if cfg.add_static_data_to_vocab:
        static_data_path = output_dir / STATIC_DATA_FN
        if static_data_path.is_file():
            with (output_dir / STATIC_DATA_FN).open("rb") as f:
                static_codes = []
                for pt_static_data in pickle.load(f).values():
                    for static_data_obj in pt_static_data.values():
                        if static_data_obj is not None:
                            for code in static_data_obj["code"]:
                                static_codes.append(code)
                static_codes = sorted(set(static_codes).difference([str(ST.DOB)]))
                list_of_words.extend(static_codes)
            # vocab.add_words(static_codes)

    codes = pl.read_csv(output_dir / cfg.code_counts_fn, columns="code")["code"].to_list()
    list_of_words.extend(codes)
    list_of_words = sorted(list(set(list_of_words)))
    vocab.add_words(list_of_words)

    vocab.dump(output_dir)


if __name__ == "__main__":
    run()