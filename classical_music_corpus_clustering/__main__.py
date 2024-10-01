import fsspec
from pathlib import Path
import pandas as pd


def get_df_by_position_with_name(position, dfs):
    key = list(dfs.keys())[position]
    return dfs[key]["expanded"][0][1].assign(piece=key)


if __name__ == "__main__":

    destination = Path(".") / "corpus"
    destination.mkdir(exist_ok=True, parents=True)
    fs = fsspec.filesystem("github", org="dcmlab", repo="mozart_piano_sonatas")
    fs.get(fs.ls("harmonies/"), destination.as_posix())
    fs.get("metadata.tsv", (destination / "metadata.tsv").as_posix())

    corpus = ms3.Corpus("corpus/")

    corpus.parse_tsv()
    dfs = corpus.get_dataframes()
    adf = get_df_by_position_with_name(0, dfs)

    cdf = pd.concat(
        [
            get_df_by_position_with_name(i, dfs).assign(composer="Mozart")
            for i in range(len(dfs))
        ]
    )

    tone_frequency_by_piece = (
        cdf.loc[:, "piece composer chord_tones".split()]
        .set_index("piece composer".split())
        .chord_tones.explode()
        .groupby(level=(0, 1))
        .value_counts()
        .astype("Int64")
        .unstack()
        .fillna(0)
    )
