import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional

from load_data import TEXT_COL


def get_tf_idf_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes TF-IDF embeddings for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the TF-IDF vectors (one per row)
        and the feature‐to‐index mapping.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df[column_name].astype(str))

    return tfidf_matrix.toarray().tolist()


def get_word2vec_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes Word2Vec embeddings for a specified column in a DataFrame (one vector per word in the corpus).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the Word2Vec word‐vectors (one per word)
        and the word‐to‐index mapping.
    """
    from gensim.models import Word2Vec

    # Tokenize each sentence into a list of words
    sentences = [str(text).split() for text in df[column_name].astype(str)]
    # Train a Word2Vec model on those sentences
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

    # Extract the vector for each word in the same order
    vectors = [model.wv[word].tolist() for word in model.wv.index_to_key]

    return vectors


def get_doc2vec_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes Doc2Vec embeddings for a specified column in a DataFrame (one vector per document).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the Doc2Vec document‐vectors
        (one per row in df) and a mapping from document tag to index.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    # We need to create TaggedDocument objects so that Doc2Vec learns a vector for each row.
    tagged_docs = [
        TaggedDocument(words=str(text).split(), tags=[str(i)])
        for i, text in enumerate(df[column_name].astype(str))
    ]
    # Train a Doc2Vec model
    model = Doc2Vec(tagged_docs, vector_size=100, window=5, min_count=1, workers=4)

    # Extract the vector for each tag in the same order
    vectors = [model.dv[tag].tolist() for tag in model.dv.index_to_key]

    return vectors


def get_bert_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes BERT embeddings for a specified column in a DataFrame (one vector per document).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the BERT sentence‐vectors
        (one per row in df) and a dummy token‐to‐index mapping (dimension indices).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Encode() returns a [n_rows × embedding_dim] NumPy array
    embeddings = model.encode(
        df[column_name].astype(str).tolist(), convert_to_tensor=False
    )

    return embeddings.tolist()


def get_glove_embedding(
    df, column_name: str, max_length: int = 100, embedding_dim: int = 100
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes fixed‐size GloVe embeddings for a specified column in a DataFrame.
    Each sentence is padded/truncated to max_length tokens, and each token is represented by
    its 100‐dimensional GloVe vector. The final output is a flattened list of length
    (max_length * embedding_dim) per row.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.
        max_length (int, optional): The maximum number of tokens per sentence. Defaults to 100.
        embedding_dim (int, optional): The dimensionality of the GloVe vectors. Defaults to 100.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
            1) A list of flattened GloVe embeddings (one list of length max_length*embedding_dim per row).
            2) A feature‐to‐index mapping of the form {"feature_0": 0, "feature_1": 1, ..., "feature_{(max_length*embedding_dim)-1}": (max_length*embedding_dim)-1}.
    """
    import numpy as np
    from torchtext.vocab import GloVe

    # Load pre‐trained GloVe vectors (6B tokens, 100‐dim)
    glove = GloVe(name="6B", dim=embedding_dim)

    def sentence_embedding(sentence: str) -> List[float]:
        """
        Converts a single sentence to a fixed‐size, flattened GloVe representation.
        Pads with zeros or truncates so that there are exactly max_length tokens.
        """
        tokens = sentence.split()
        num_tokens = min(len(tokens), max_length)
        # Initialize a (max_length × embedding_dim) zero‐matrix
        matrix = np.zeros((max_length, embedding_dim), dtype=float)

        for i in range(num_tokens):
            w = tokens[i]
            if w in glove.stoi:
                # Retrieve the GloVe vector (as a torch.Tensor) and convert to NumPy
                matrix[i] = glove.vectors[glove.stoi[w]].numpy()

        # Flatten to a single list of length max_length * embedding_dim
        return matrix.flatten().tolist()

    # Apply to every row in the DataFrame column
    embeddings_per_row = [
        sentence_embedding(str(text)) for text in df[column_name].astype(str)
    ]

    return embeddings_per_row


def get_bert_cls_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes BERT [CLS] token embeddings for a specified column in a DataFrame (one 768‐dim vector per document).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the BERT [CLS] embeddings
        (one 768‐dim vector per row in df) and a mapping from "cls_i" to dimension index.
    """
    import torch
    from transformers import BertTokenizer, BertModel

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    embeddings_list: List[List[float]] = []

    for text in df[column_name].astype(str):
        # Tokenize input sentence and convert to tensor (with special tokens, truncated to max_length=512)
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]  # shape: [1, seq_len]
        attention_mask = encoded["attention_mask"]

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            # outputs[0] is the last hidden state: shape [1, seq_len, hidden_size]
            # We take the hidden state corresponding to [CLS] at position 0
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape [1, hidden_size]
            cls_vector = cls_embedding.squeeze(0).tolist()  # list of length 768

        embeddings_list.append(cls_vector)

    return embeddings_list


def get_gte_small_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes embeddings using the "thenlper/gte-small" SentenceTransformer model
    for a specified column in a DataFrame (one vector per document).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the GTE‐small vectors
        (one per row in df) and a dummy token‐to‐index mapping (dimension indices).
    """
    from sentence_transformers import SentenceTransformer

    # Load the pre-trained "thenlper/gte-small" model
    model = SentenceTransformer("thenlper/gte-small")
    # Encode all rows in the specified column; show_progress_bar=True displays progress
    embeddings = model.encode(
        df[column_name].astype(str).tolist(),
        show_progress_bar=True,
        convert_to_tensor=False,
    )

    return embeddings.tolist()


def get_openai_embedding(
    df, column_name: str
) -> Tuple[List[List[float]], Dict[str, int]]:
    """
    Computes OpenAI embeddings for a specified column in a DataFrame (one vector per document).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to compute embeddings for.
        model_name (str, optional): The OpenAI embedding model to use. Defaults to "text-embedding-3-small".

    Returns:
        Tuple[List[List[float]], Dict[str, int]]: A tuple containing the OpenAI embeddings
        (one vector per row in df) and a mapping from "token_i" to dimension index.
    """
    from openai import OpenAI

    # Convert the column to a list of strings
    docs = df[column_name].astype(str).tolist()

    # Initialize the OpenAI client (ensure OPENAI_API_KEY is set in your environment)
    client = OpenAI()
    response = client.embeddings.create(input=docs, model="text-embedding-3-small")

    # Each item in response.data has an 'embedding' attribute which is a list of floats
    embeddings = [x.embedding for x in response.data]

    return embeddings


def get_e5_embedding(df, column_name: str) -> Tuple[List[List[float]], Dict[str, int]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("intfloat/e5-small")
    embeddings = model.encode(
        df[column_name].astype(str).tolist(),
        normalize_embeddings=True,
    )
    return embeddings.tolist()


def embed(
    df: pd.DataFrame,
    embedding_type: str,
    output_path: str,
    *,
    # If you want to run UMAP after getting the raw embeddings, set this to True.
    text_column: str = TEXT_COL,
    apply_umap: bool = False,
    # UMAP-specific parameters (only used if apply_umap=True).
    umap_params: Optional[Dict[str, Any]] = None,
    # Any remaining kwargs go to the chosen embedding function.
    **kwargs: Any,
) -> Tuple[np.ndarray, Dict[str, int], str]:
    """
    1) Computes “raw” embeddings according to embedding_type.
    2) If apply_umap=True, reduces those embeddings with UMAP.
    3) Builds a DataFrame [timestamp, embed_0,…,embed_{d-1}], saves to CSV.
    4) Returns (final_embeddings_array, feature_mapping, saved_csv_path).

    Args:
        df: DataFrame containing at least the text and timestamp columns.
        text_column: Name of the column to embed.
        embedding_type: One of {"tfidf", "word2vec", "doc2vec", "bert", "glove", "bert_cls", "gte_small"}.
        output_path: Path (including filename) where the CSV should be written.
        apply_umap: If True, run UMAP on the embeddings returned by the chosen embedding function.
        umap_params: Dict of UMAP hyperparameters. Possible keys:
            - n_components (int, default=5)
            - min_dist (float, default=0.0)
            - metric (str, default="cosine")
            - random_state (int, default=42)
        **kwargs: Passed directly into the underlying embedding function (e.g. max_length for GloVe).

    Returns:
        embeddings_array: np.ndarray of shape (n_samples, final_dim).
                          (If apply_umap=False, final_dim = original embedding dim.)
                          (If apply_umap=True, final_dim = umap_params["n_components"].)
        feature_mapping: Dict mapping feature‐names (or "umap_i") → column index.
        saved_csv_path: The filepath where the CSV was written.
    """
    # 1. Lookup table: embedding_type -> function
    embedding_lookup: Dict[
        str, Callable[..., Tuple[List[List[float]], Dict[str, int]]]
    ] = {
        "tfidf": get_tf_idf_embedding,
        "word2vec": get_word2vec_embedding,
        "doc2vec": get_doc2vec_embedding,
        "bert": get_bert_embedding,
        "glove": get_glove_embedding,
        "bert_cls": get_bert_cls_embedding,
        "gte_small": get_gte_small_embedding,
        "openai": get_openai_embedding,
        "e5": get_e5_embedding,
    }

    embedding_func = embedding_lookup.get(embedding_type.lower())
    if embedding_func is None:
        raise ValueError(
            f"Unknown embedding_type '{embedding_type}'. "
            f"Valid options: {list(embedding_lookup.keys())}."
        )

    # 2. Call the chosen embedding function
    #    Each returns (List[List[float]], Dict[str,int])
    embeddings_list = embedding_func(df, text_column, **kwargs)
    embeddings_array = np.array(embeddings_list)  # shape = (n_samples, raw_dim)

    # 3. Optionally reduce with UMAP
    if apply_umap:
        # Set defaults if not provided
        umap_params = umap_params or {}
        n_components = umap_params.get("n_components", 5)
        min_dist = umap_params.get("min_dist", 0.0)
        metric = umap_params.get("metric", "cosine")
        random_state = umap_params.get("random_state", 42)

        from umap import UMAP as _UMAP

        umap_model = _UMAP(
            n_components=n_components,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        reduced = umap_model.fit_transform(
            embeddings_array
        )  # shape = (n_samples, n_components)
        embeddings_array = np.array(reduced)

    # 4. Build DataFrame with timestamps + embed_0…embed_{d-1}
    _, embedding_dim = embeddings_array.shape
    embed_cols = [f"embed_{i}" for i in range(embedding_dim)]
    result_df = pd.DataFrame(embeddings_array, columns=embed_cols)

    # 5. Ensure directory exists, then save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    return embeddings_array
