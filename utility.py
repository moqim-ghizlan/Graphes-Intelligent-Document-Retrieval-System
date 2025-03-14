"""
==================================================
Utility Functions for Data Analysis and Processing
==================================================
This script provides a collection of utility functions for:
- Loading and analyzing dataset characteristics.
- Preprocessing text data for machine learning applications.
- Computing word frequencies and text similarities.
- Clustering analysis and evaluation.
- Graph-based analysis, including adjacency matrix operations and subgraph visualization.
- Author prediction based on text representation techniques.
- Visualization functions for results interpretation.

Author: JUILLARD Thibaut and GHIZLAN Moqim


@@ ChatGPT and Github copilot were used to write this code specifically to generate the documentation and the comments.
   Some of the functions were taken from ./src/files/<file_name>.py provided by J.V but were modified to fit the current project.

"""



import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from itertools import combinations



def fetch_data(path):
    """
    Load a dataset from a CSV file.

    Parameters:
    - path (str): The file path to the dataset.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the loaded dataset.
    """
    return pd.read_csv(path, sep="\t")


def data_analyse(data):
    """
    Perform a comprehensive exploratory data analysis (EDA) on a given dataset.

    Parameters:
    - data (pd.DataFrame): The dataset to be analyzed.

    This function prints:
    - The first few rows of the dataset.
    - The shape of the dataset (number of rows and columns).
    - Descriptive statistics for numerical columns.
    - Information about column data types and missing values.
    - The count of missing values per column.
    - Length statistics for textual columns ('abstract', 'title', 'venue').
    - The number of unique articles and authors.
    - The distribution of publication years (if available).
    - The distribution of citation counts (if available).
    - The number of unique values per column.
    - A count of missing or empty abstracts.
    """

    # Display the first few rows of the DataFrame
    print("Preview of the first few rows:")
    print(data.head())

    # Display the shape of the DataFrame
    print("\nSize of the DataFrame (rows, columns):")
    print(data.shape)

    # Display descriptive statistics for numerical columns
    print("\nDescriptive statistics for numerical columns:")
    print(data.describe())

    # Display information about data types and missing values
    print("\nColumn information (data types and missing values):")
    print(data.info())

    # Count missing values per column
    print("\nMissing values per column:")
    print(data.isnull().sum())

    # Compute length statistics for textual columns if they exist
    if "abstract" in data.columns:
        data["abstract_length"] = data["abstract"].apply(lambda x: len(str(x)))
        print("\nStatistics for 'abstract' column length:")
        print(
            f"Min: {data['abstract_length'].min()}, Mean: {data['abstract_length'].mean()}, Max: {data['abstract_length'].max()}"
        )

    if "title" in data.columns:
        data["title_length"] = data["title"].apply(lambda x: len(str(x)))
        print("\nStatistics for 'title' column length:")
        print(
            f"Min: {data['title_length'].min()}, Mean: {data['title_length'].mean()}, Max: {data['title_length'].max()}"
        )

    if "venue" in data.columns:
        data["venue_length"] = data["venue"].apply(lambda x: len(str(x)))
        print("\nStatistics for 'venue' column length:")
        print(
            f"Min: {data['venue_length'].min()}, Mean: {data['venue_length'].mean()}, Max: {data['venue_length'].max()}"
        )

    # Compute the number of unique articles and authors
    num_articles = data.shape[0]
    num_unique_authors = data["authors"].nunique() if "authors" in data.columns else 0
    print(f"\nNumber of articles: {num_articles}")
    print(f"Number of unique authors: {num_unique_authors}")

    # Display the distribution of publication years if available
    if "year" in data.columns:
        print("\nDistribution of publication years:")
        print(data["year"].value_counts())

    # Display the distribution of citation counts if available
    if "n_citation" in data.columns:
        print("\nDistribution of citation counts:")
        print(data["n_citation"].value_counts().head(10))

    # Display the number of unique values per column
    print("\nUnique values per column:")
    print(data.nunique())

    # Count missing and empty abstracts
    missing_abstracts = data["abstract"].isna().sum()
    empty_abstracts = (data["abstract"].str.strip() == "").sum()

    print(f"\nTotal number of documents: {len(data)}")
    print(f"Documents with missing abstracts (NaN): {missing_abstracts}")
    print(f"Documents with empty abstracts (''): {empty_abstracts}")

    # Display distribution of text length for key text columns
    print("\nDistribution of text length (abstract, title, venue):")
    if "abstract_length" in data.columns:
        print(
            f"Abstract - Min: {data['abstract_length'].min()}, Mean: {data['abstract_length'].mean()}, Max: {data['abstract_length'].max()}"
        )
    if "title_length" in data.columns:
        print(
            f"Title - Min: {data['title_length'].min()}, Mean: {data['title_length'].mean()}, Max: {data['title_length'].max()}"
        )
    if "venue_length" in data.columns:
        print(
            f"Venue - Min: {data['venue_length'].min()}, Mean: {data['venue_length'].mean()}, Max: {data['venue_length'].max()}"
        )



def preprocess_text(text, stemmer=None):
    """
    Preprocesses a given text by applying tokenization, stopword removal, and optional stemming.

    Parameters:
    - text (str): The input text to preprocess.
    - stemmer (nltk.stem.SnowballStemmer, optional): A stemmer to apply word stemming (default: None).

    Returns:
    - str: The cleaned and preprocessed text.
    """

    if not isinstance(text, str):  # Convert non-string values to an empty string
        text = ""

    # Tokenization and stopword removal
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]

    # Apply stemming if provided
    if stemmer:
        words = [stemmer.stem(word) for word in words]

    return " ".join(words)

def word_frequencies(df, column):
    """
    Computes word frequencies in a given text column of a DataFrame.

    Parameters:
    - df (pd.DataFrame): The dataset containing the text data.
    - column (str): The name of the column containing text data.

    Returns:
    - pd.DataFrame: A DataFrame with two columns: 'Word' and 'Frequency', sorted in descending order.
    """

    # Combine all text entries into a single string
    text = " ".join(df[column].dropna().astype(str))

    # Tokenize words and count their occurrences
    words = text.lower().split()
    word_count = Counter(words)

    # Convert to DataFrame and sort by frequency
    freq_df = pd.DataFrame(word_count.items(), columns=["Word", "Frequency"])
    return freq_df.sort_values(by="Frequency", ascending=False)

def compute_accuracy(df, text_column, label_column, vectorizer_type="tfidf"):
    """
    Computes the classification accuracy using either TF-IDF or Count Vectorization with Naïve Bayes.

    Parameters:
    - df (pd.DataFrame): The dataset containing text and labels.
    - text_column (str): The name of the column containing text data.
    - label_column (str): The name of the column containing class labels.
    - vectorizer_type (str, optional): Type of vectorization ('tfidf' for TF-IDF, 'count' for CountVectorizer). Default is 'tfidf'.

    Returns:
    - float: The classification accuracy of the Naïve Bayes model.
    """

    # Select vectorizer based on user input
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()

    # Transform the text data into numerical feature vectors
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train a Naïve Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute accuracy score
    acc = accuracy_score(y_test, y_pred)
    return acc






def preprocess_query(query, stemmer=None):
    """
    Preprocesses the user query by applying tokenization, stopword removal, and stemming.

    Parameters:
    - query (str): The raw user input string.
    - stemmer (nltk.stem.SnowballStemmer, optional): A stemmer to apply word stemming (default: None).

    Returns:
    - list: A list of processed keywords after tokenization and optional stemming.
    """

    if not isinstance(query, str):  # Convert non-string values to an empty string
        query = ""

    # Tokenization and stopword removal
    stop_words = set(stopwords.words("english"))
    words = query.split()
    words = [word.lower() for word in words if word.lower() not in stop_words]

    # Apply stemming if provided
    if stemmer:
        words = [stemmer.stem(word) for word in words]

    return words  # Return the processed list of keywords


def compute_similarity(query_vector, document_matrix):
    """
    Computes cosine similarity between a query vector and all document vectors.

    Parameters:
    - query_vector (array-like): Vector representation of the user query (1, n_features).
    - document_matrix (array-like): Matrix containing all document vectors (n_docs, n_features).

    Returns:
    - tuple: Two sorted arrays:
        - sorted_scores (array): Cosine similarity scores sorted in descending order.
        - sorted_indices (array): Indices of documents sorted by relevance.
    """

    similarity_scores = cosine_similarity(query_vector, document_matrix).flatten()

    # Sort scores in descending order and retrieve corresponding indices
    sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
    sorted_scores = similarity_scores[sorted_indices]

    return sorted_scores, sorted_indices  # Return sorted similarity scores and indices


def display_topics(model, feature_names, num_words=10):
    """
    Displays the top words associated with each topic in a topic modeling model.

    Parameters:
    - model (sklearn.decomposition.LatentDirichletAllocation): The trained LDA model.
    - feature_names (list): List of feature names from the vectorizer.
    - num_words (int, optional): Number of top words to display per topic (default: 10).
    """

    for topic_idx, topic in enumerate(model.components_):
        print(f"\n**Topic {topic_idx}**:")
        print(", ".join([feature_names[i] for i in topic.argsort()[:-num_words - 1:-1]]))


def get_cooccurrences(text_series, window_size=2):
    """
    Extracts co-occurring word pairs within a given window size.

    Parameters:
    - text_series (pd.Series): Series of text documents.
    - window_size (int, optional): The number of words to consider for co-occurrence (default: 2).

    Returns:
    - Counter: A dictionary-like object with word pairs as keys and co-occurrence counts as values.
    """

    word_pairs = Counter()
    for text in text_series:
        words = text.split()
        for pair in combinations(words[:window_size], 2):  # Take words in the same window
            word_pairs[pair] += 1
    return word_pairs


def find_word_in_corpus(word, data, vectorizer, X):
    """
    Finds the most relevant documents containing a given word using cosine similarity.

    Parameters:
    - word (str): The word to search for.
    - data (pd.DataFrame): The dataset containing the document information.
    - vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The vectorizer used for text representation.
    - X (scipy.sparse matrix): The document-term matrix created using the vectorizer.

    Returns:
    - None: Prints the top 10 most relevant documents containing the word.
    """

    # Convert the word into a vector representation
    word_vec = vectorizer.transform([word])

    # Compute similarity between the word vector and all document vectors
    similarities = cosine_similarity(word_vec, X)

    # Get the indices of the top 10 most relevant documents
    top_docs = similarities.argsort()[0][-10:][::-1]  # Top 10 documents sorted by relevance

    print(f"\nDocuments containing '{word}':\n")
    for doc_id in top_docs:
        print(f"{data.loc[doc_id, 'title']}")
        print(f" Abstract: {data.loc[doc_id, 'abstract'][:300]}...")  # Display only first 300 chars
        print("-" * 80)


def top_words_per_cluster(data, text_column, cluster_column, top_n=10):
    """
    Extracts the most frequent words for each cluster.

    Parameters:
    - data (pd.DataFrame): The dataset containing text and cluster labels.
    - text_column (str): The column containing the processed text.
    - cluster_column (str): The column containing cluster labels.
    - top_n (int, optional): Number of top words to extract per cluster (default: 10).

    Returns:
    - dict: A dictionary where keys are cluster IDs and values are lists of top words with frequencies.
    """

    cluster_words = {}

    for cluster_id in sorted(data[cluster_column].unique()):
        cluster_texts = " ".join(data[data[cluster_column] == cluster_id][text_column])
        words = cluster_texts.split()
        word_counts = Counter(words)
        cluster_words[cluster_id] = word_counts.most_common(top_n)

    return cluster_words


def clean_and_tokenize(text):
    """
    Cleans and tokenizes a given text by removing special characters, lowercasing, and filtering stopwords.

    Parameters:
    - text (str): The input text to be processed.

    Returns:
    - list: A list of cleaned words.
    """

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters except spaces
    words = re.findall(r'\b\w{3,}\b', text)  # Extract words with at least 3 characters
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return words


def top_tfidf_terms_per_cluster(data, vectorizer, X_vectorized, cluster_column, top_n=10):
    """
    Extracts the top TF-IDF terms for each cluster.

    Parameters:
    - data (pd.DataFrame): The dataset containing text and cluster labels.
    - vectorizer (sklearn.feature_extraction.text.TfidfVectorizer): The trained TF-IDF vectorizer.
    - X_vectorized (sparse matrix): The TF-IDF transformed text data.
    - cluster_column (str): The column containing cluster labels.
    - top_n (int, optional): Number of top terms to extract per cluster (default: 10).

    Returns:
    - dict: A dictionary where keys are cluster IDs and values are lists of top TF-IDF terms with scores.
    """

    clusters = data[cluster_column].unique()
    top_terms = {}

    for cluster in clusters:
        cluster_indices = data[data[cluster_column] == cluster].index
        cluster_vector = X_vectorized[cluster_indices].mean(axis=0)

        terms_scores = zip(vectorizer.get_feature_names_out(), np.asarray(cluster_vector).flatten())
        sorted_terms = sorted(terms_scores, key=lambda x: x[1], reverse=True)

        top_terms[cluster] = sorted_terms[:top_n]

    return top_terms


def central_documents_per_cluster(data, X_vectorized, cluster_column, num_docs=5):
    """
    Identifies the most central documents for each cluster based on their distance to the cluster center.

    Parameters:
    - data (pd.DataFrame): The dataset containing document titles and abstracts.
    - X_vectorized (sparse matrix): The vectorized document representations.
    - cluster_column (str): The column containing cluster labels.
    - num_docs (int, optional): Number of central documents to return per cluster (default: 5).

    Returns:
    - dict: A dictionary where keys are cluster IDs and values are DataFrames containing central documents.
    """

    central_docs = {}
    clusters = data[cluster_column].unique()

    for cluster in clusters:
        cluster_indices = data[data[cluster_column] == cluster].index
        cluster_vectors = X_vectorized[cluster_indices].toarray()  # Convert to NumPy array

        # Compute the average cluster center
        cluster_center = cluster_vectors.mean(axis=0)
        distances = pairwise_distances(cluster_vectors, cluster_center.reshape(1, -1)).flatten()

        # Select the most central documents
        central_doc_indices = cluster_indices[np.argsort(distances)[:num_docs]]
        central_docs[cluster] = data.loc[central_doc_indices, ['title', 'abstract']]

    return central_docs


def graph_to_adjacency_matrix(graph):
    """
    Converts a NetworkX graph to an adjacency matrix.

    Parameters:
    - graph (networkx.Graph): The input graph.

    Returns:
    - np.ndarray: An adjacency matrix representing the graph.
    """

    num_nodes = len(graph.nodes)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for u, v in graph.edges:
        adjacency_matrix[u, v] = 1
        adjacency_matrix[v, u] = 1  # Assuming an undirected graph

    return adjacency_matrix



def check_adjacency_consistency(G, adj_matrix):
    """
    Checks the consistency between the adjacency matrix and the NetworkX graph.

    Parameters:
    - G (networkx.Graph): The original graph.
    - adj_matrix (np.ndarray): The adjacency matrix representation of the graph.

    Prints:
    - Confirmation if all edges match correctly.
    - Example of mismatched edges if any inconsistencies are found.
    """

    mismatch_edges = []
    for u, v in G.edges():
        if adj_matrix[u, v] != 1 or adj_matrix[v, u] != 1:
            mismatch_edges.append((u, v))

    if len(mismatch_edges) == 0:
        print("All edges match correctly in the adjacency matrix.")
    else:
        print(f"Mismatch found in {len(mismatch_edges)} edges. Example:", mismatch_edges[:10])



def find_authors_tfidf(article_vector, author_rep_tfidf, top_k=5):
    """
    Finds the best-matching authors based on TF-IDF similarity.

    Parameters:
    - article_vector (np.ndarray): Vector representation of the article.
    - author_rep_tfidf (dict): Dictionnary mapping authors to their TF-IDF vectors.
    - top_k (int, optional): How many top authors to return (default: 5).

    Returns:
    - list of tuples: List of authors with similarity, sorted in descending order.
    """

    # Calculate similarity between article vector and each author's vector
    similarities = {
        author: cosine_similarity(article_vector.reshape(1, -1), vec.reshape(1, -1))[0, 0]
        for author, vec in author_rep_tfidf.items()
    }

    # Return top k authors sorted by similarity
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]



def find_authors_doc2vec(article_vector, author_rep_doc2vec, top_k=5):
    """
    Finds the best-matching authors based on Doc2Vec similarity.

    Parameters:
    - article_vector (np.ndarray): Vector representation of the article.
    - author_rep_doc2vec (dict): Dictionnary mapping authors to their Doc2Vec vectors.
    - top_k (int, optional): How many top authors to return (default: 5).

    Returns:
    - list of tuples: List of authors with similarity, sorted in descending order.
    """

    # Calculate similarity between article vector and each author's vector
    similarities = {
        author: cosine_similarity(article_vector.reshape(1, -1), vec.reshape(1, -1))[0, 0]
        for author, vec in author_rep_doc2vec.items()
    }

    # Return top k authors sorted by similarity
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]



def evaluate_author_prediction(data, X_tfidf, doc_vectors, author_rep_tfidf, author_rep_doc2vec, method="tfidf", top_k=5):
    """
    Evaluate how accurately the true authors are predicted based on TF-IDF or Doc2Vec.

    Parameters:
    - data (pd.DataFrame): DataFrame with at least an 'authors' column.
    - X_tfidf (sparse matrix): TF-IDF vectors for each article.
    - doc_vectors (list or np.ndarray): Doc2Vec vectors for each article.
    - author_rep_tfidf (dict): Dictionnary mapping authors to TF-IDF vectors.
    - author_rep_doc2vec (dict): Dictionnary mapping authors to Doc2Vec vectors.
    - method (str, optional): Method used for similarity ('tfidf' or 'doc2vec'). Default is 'tfidf'.
    - top_k (int, optional): Number of top predictions to consider (default: 5).

    Returns:
    - float: Average rank position of true authors among predictions (lower is better).
    """

    ranks = []

    for idx, row in data.iterrows():
        # Extract true authors of current article
        true_authors = row["authors"].split(",") if pd.notna(row["authors"]) else []

        # Predict authors depending on chosen method
        if method == "tfidf":
            predicted_authors = find_authors_tfidf(
                X_tfidf[idx].toarray().flatten(), author_rep_tfidf, top_k=top_k
            )
        else:
            predicted_authors = find_authors_doc2vec(
                doc_vectors[idx], author_rep_doc2vec, top_k=top_k
            )

        predicted_names = [a[0] for a in predicted_authors]

        # Check position of each true author in predictions
        for author in true_authors:
            if author in predicted_names:
                # Rank is the position + 1
                ranks.append(predicted_names.index(author) + 1)
            else:
                # If author not in top_k predictions, assign top_k + 1 as rank
                ranks.append(top_k + 1)

    # Return average rank across all articles/authors
    return np.mean(ranks)
