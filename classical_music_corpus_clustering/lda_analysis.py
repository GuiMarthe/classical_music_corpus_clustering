import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from classical_music_corpus_clustering.__main__ import extract_corpora, build_trigrams


def calculate_lda_icl(X, lda_model):
    """
    Calculate the Integrated Completed Likelihood (ICL) criterion for a fitted LDA model.

    Parameters:
    -----------
    X : array-like of shape (n_documents, n_words)
        The document-term matrix
    lda_model : LatentDirichletAllocation
        A fitted LDA model

    Returns:
    --------
    icl : float
        The ICL score. Lower values indicate better models.
    """
    n_documents, vocabulary_size = X.shape
    n_topics = lda_model.n_components

    # Get document-topic and topic-word distributions
    doc_topic_distr = lda_model.transform(X)  # shape: (n_documents, n_topics)

    # Calculate log likelihood
    log_likelihood = lda_model.score(X) * n_documents

    # Calculate number of free parameters
    # Parameters: topic-word distributions + document-topic distributions - 1 for sum-to-one constraints
    n_parameters = (n_topics * vocabulary_size - vocabulary_size) + (
        n_documents * n_topics - n_documents
    )

    # Calculate BIC
    bic = -2 * log_likelihood + n_parameters * np.log(n_documents)

    # Calculate entropy term for document-topic assignments
    # Only include non-zero probabilities in log calculation
    mask = doc_topic_distr > np.finfo(doc_topic_distr.dtype).tiny
    entropy = -np.sum(doc_topic_distr[mask] * np.log(doc_topic_distr[mask]))

    # Calculate ICL
    icl = bic + 2 * entropy

    return icl


def find_optimal_topics_icl(X, max_topics, random_state=42):
    """
    Find the optimal number of topics using ICL criterion.

    Parameters:
    -----------
    X : array-like of shape (n_documents, n_words)
        The document-term matrix
    max_topics : int
        Maximum number of topics to try
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    optimal_n_topics : int
        Optimal number of topics
    icl_scores : list
        List of ICL scores for each number of topics
    """
    icl_scores = []

    for n_topics in range(2, max_topics + 1):  # Start from 2 topics
        # Fit LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            n_jobs=-1,  # Use all available cores
            max_iter=20,  # Increase if needed
            learning_method="batch",
        ).fit(X)

        # Calculate ICL
        icl = calculate_lda_icl(X, lda)
        icl_scores.append(icl)

    # Find optimal number of topics
    optimal_n_topics = (
        np.argmin(icl_scores) + 2
    )  # Add 2 because we started from 2 topics

    return optimal_n_topics, icl_scores


def print_top_words(model, feature_names, n_top_words=10):
    """
    Function to display top trigrams for each topic
    """
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")


def plot_topics_search(X, vectorizer):
    max_topics = 60
    optimal_n_topics, icl_scores = find_optimal_topics_icl(X, max_topics)

    # Plot ICL scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_topics + 1), icl_scores, "bo-")
    plt.xlabel("Number of Topics")
    plt.ylabel("ICL Score")
    plt.title("ICL Score vs Number of Topics")
    plt.axvline(
        x=optimal_n_topics,
        color="r",
        linestyle="--",
        label=f"Optimal topics: {optimal_n_topics}",
    )
    plt.legend()
    plt.grid(True)

    print(f"Optimal number of topics according to ICL: {optimal_n_topics}")
    print(f"ICL scores: {icl_scores}")

    # Fit LDA with optimal number of topics and display results
    optimal_lda = LatentDirichletAllocation(
        n_components=optimal_n_topics, random_state=42
    ).fit(X)

    print("\nTop words in each topic:")
    print_top_words(optimal_lda, vectorizer.get_feature_names_out())


def get_base_data():
    cdf = extract_corpora()
    chord_sequence = cdf.set_index("composer piece".split()).loc[:, "chord".split()]
    tri = build_trigrams(chord_sequence, "chord").reset_index(0, drop=True)
    top_1200_tri = tri.value_counts(normalize=False).reset_index().head(1200)
    top_mask = tri.isin(top_1200_tri.chord)
    filtered_tri = tri[top_mask]
    return filtered_tri


if __name__ == "__main__":
    trigrams = get_base_data()
    vocab = set(trigrams)
    documents = trigrams.groupby(level=(0, 1)).apply(lambda x: " ".join(x))

    vectorizer = CountVectorizer(
        vocabulary=vocab, token_pattern=r"[^\s]+", lowercase=False, binary=False
    )
    X = vectorizer.fit_transform(documents.tolist())

    plot_topics_search(X, vectorizer)

    vectorizer = CountVectorizer(
        vocabulary=vocab, token_pattern=r"[^\s]+", lowercase=False, binary=True
    )
    X = vectorizer.fit_transform(documents.tolist())

    plot_topics_search(X, vectorizer)

    vectorizer = TfidfVectorizer(
        vocabulary=vocab, token_pattern=r"[^\s]+", lowercase=False
    )
    X = vectorizer.fit_transform(documents.tolist())

    plot_topics_search(X, vectorizer)
