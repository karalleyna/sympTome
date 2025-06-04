import spacy
from keybert import KeyBERT
from fuzzywuzzy import process

import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load .env file
load_dotenv()

# Retrieve token and login
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token)


def extract_keywords_textrank(df, column):
    # load a spaCy model, depending on language, scale, etc.
    nlp_textrank = spacy.load("en_core_web_sm")

    # add PyTextRank to the spaCy pipeline
    nlp_textrank.add_pipe("textrank")

    doc = nlp_textrank(text)
    doc_keywords = [
        keyword.text for keyword in doc._.phrases if len(keyword.text.split()) <= 3
    ]
    deduplicated_doc_keywords = list(process.dedupe(doc_keywords, threshold=70))
    final_keywords = ", ".join(deduplicated_doc_keywords[:6])
    return final_keywords


from sentence_transformers import SentenceTransformer

# Outside function: load once, reuse
kw_model = KeyBERT(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"))


def extract_keywords_keybert(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3))
    doc_keywords = [keyword[0] for keyword in keywords]
    deduplicated_doc_keywords = list(process.dedupe(doc_keywords, threshold=70))
    final_keywords = deduplicated_doc_keywords[:6]
    final_keywords = " ".join(final_keywords) if final_keywords else ""
    return final_keywords


def explode_column(df, column):
    # Step 1: Filter out rows with empty lists
    df_filtered = df[df[column].map(len) > 0].copy()

    # Step 2: Explode the column so each list element becomes its own row
    df_exploded = df_filtered.explode(column).reset_index(drop=True)
    return df_exploded


def extract_keywords(df, columns, method):
    extract_function = {
        "keybert": extract_keywords_keybert,
        "textrank": extract_keywords_textrank,
    }.get(method, None)

    if extract_function is None:
        raise ValueError(f"Invalid method: {method}. Choose 'keybert' or 'textrank'.")

    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        df[f"{column}_clean"] = df[column].astype(str)
        df[column] = df[column].apply(extract_function)
        # df = explode_column(df, column)
    return df
