import pandas as pd
from datasets import load_dataset


def load_datasets():
    """
    Only consider instances for which a FR translation is available.
    """
    en_ds = load_dataset("mteb/mtop_domain", "en")
    fr_ds = load_dataset("mteb/mtop_domain", "fr")

    def _merge(en_df, fr_df):
        return en_df.merge(fr_df[["id", "text"]], on="id", how="left", suffixes=("_en", "_fr"))\
            .dropna().reset_index(drop=True)

    split_dfs = {}
    for split in ["train", "validation", "test"]:
        en_df = pd.DataFrame(en_ds[split])
        fr_df = pd.DataFrame(fr_ds[split])
        split_dfs[split] = _merge(en_df, fr_df)

    cols = ["id", "text_en", "text_fr", "label", "label_text"]
    return split_dfs["train"][cols], split_dfs["validation"][cols], split_dfs["test"][cols]
