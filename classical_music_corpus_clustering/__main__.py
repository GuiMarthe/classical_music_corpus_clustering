import fsspec
from pathlib import Path
import pandas as pd
import ms3
from dataclasses import dataclass
from typing import Optional
from sklearn.decomposition import LatentDirichletAllocation


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
class CorpusMD:
    repo: str
    destination: Path
    corpus_ms3 = Optional[ms3.corpus.Corpus]

    @property
    def composer(self) -> str:
        return self.repo.split("_")[0].title()

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


def extract_corpora():
    repos = ["mozart_piano_sonatas", "beethoven_piano_sonatas", "debussy_suite_bergamasque"]

    corpora = [CorpusMD(repo, make_destination(repo)) for repo in repos]

    if not Path("corpus/").exists():
        download_corpora(corpora)

    cdf = pd.concat([extract_dataframes(corpus) for corpus in corpora])
    return cdf


def build_trigrams(frame, col):
    trigrams = frame.groupby(level=(0, 1), as_index=False).apply(
        lambda df: df[col]
        .str.cat(df.chord.shift(-1), sep="-")
        .str.cat(df[col].shift(-2), sep="-")
    ).dropna()
    return trigrams


def freq_top_trigrams():
    cdf = extract_corpora()
    chord_sequence = cdf.set_index('composer piece'.split()).loc[:, 'chord'.split()]
    tri = build_trigrams(chord_sequence, 'chord').reset_index(0, drop=True)
    top_1200_tri = tri.value_counts(normalize=False).reset_index().head(1200)
    top_mask = tri.isin(top_1200_tri.chord)
    by_frequency_by_trigrams = tri[top_mask].groupby(level=(0,1)).value_counts()
    return by_frequency_by_trigrams


def get_lda_labels_prob(X, topics = 10):
    labels = LatentDirichletAllocation(n_components=topics).fit_transform(X)
    return pd.DataFrame(labels, index = X.index)


if __name__ == "__main__":
    freq_top_trigrams()
