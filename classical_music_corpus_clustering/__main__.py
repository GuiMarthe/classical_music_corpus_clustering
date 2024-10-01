import fsspec
from pathlib import Path
import pandas as pd
import ms3
from dataclasses import dataclass
from typing import Optional


def get_df_by_position_with_name(position, dfs) -> pd.DataFrame:
    key = list(dfs.keys())[position]
    return dfs[key]["expanded"][0][1].assign(piece=key)


def download_corpus_harmony(repo: str, destination: Path) -> None:
    fs = fsspec.filesystem("github", org="dcmlab", repo=repo)
    fs.get(fs.ls("harmonies/"), destination.as_posix())
    fs.get("metadata.tsv", (destination / "metadata.tsv").as_posix())


def make_destination(repo: str) -> Path:
    destination = Path(".") / "corpus" / repo
    destination.mkdir(exist_ok=True, parents=True)
    return destination


@dataclass
class CorpusMD():
    repo: str
    destination: Path
    corpus_ms3 = Optional[ms3.corpus.Corpus]

    @property
    def composer(self) -> str:
        return self.repo.split('_')[0].title()

    def get_dataframes(self):
        self.corpus_ms3.parse_tsv()
        return self.corpus_ms3.get_dataframes()


def download_corpora(corpora: list[CorpusMD]):
    for corpus in corpora:
        download_corpus_harmony(corpus.repo, corpus.destination)


def extract_dataframes(corpus: CorpusMD):
    corpus.corpus_ms3 = ms3.Corpus(corpus.destination.as_posix())
    dfs = corpus.get_dataframes()
    cdf = pd.concat(
        [
            get_df_by_position_with_name(i, dfs).assign(composer=corpus.composer)
            for i in range(len(dfs))
        ]
    )
    return cdf


def main():
    repos = ['mozart_piano_sonatas', 'beethoven_piano_sonatas']

    corpora = [CorpusMD(repo, make_destination(repo)) for repo in repos]

    if not Path('corpus/').exists():
        download_corpora(corpora)

    cdf = pd.concat([extract_dataframes(corpus) for corpus in corpora])

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

    return tone_frequency_by_piece



if __name__ == "__main__":
    main()
