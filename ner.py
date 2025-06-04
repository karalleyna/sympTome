import pandas as pd
from typing import List, Dict, Optional, Callable, Union
import os
from load_data import TEXT_COL
import openai


# Ensure OPENAI_API_KEY is set in environment
if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------
# 1) BERT-based NER
# ------------------
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def initialize_bert_pipeline(
    model_name: str = "dslim/distilbert-NER",
) -> Callable[[List[str]], List[List[Dict]]]:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
    model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT")
    bert_pipe = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    return bert_pipe


# -----------------------
# 2) Stanford NER Setup
# -----------------------
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
from nltk.internals import find_jars_within_path

_stanford_tagger = StanfordNERTagger("english.all.3class.distsim.crf.ser.gz")
stanford_dir = _stanford_tagger._stanford_jar.rpartition("/")[0]
stanford_jars = find_jars_within_path(stanford_dir)
_stanford_tagger._stanford_jar = ":".join(stanford_jars)


def initialize_stanford_ner() -> Callable[[List[str]], List[List[tuple]]]:
    def stanford_ner_batch(texts: List[str]) -> List[List[tuple]]:
        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append([])
                continue
            tokens = text.split()
            results.append(_stanford_tagger.tag(tokens))
        return results

    return stanford_ner_batch


# -------------------------
# 3) Hybrid CyberBERT NER
# -------------------------
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import pipeline
from preprocess import detect_and_preserve_entities

global_cyber_tokenizer = AutoTokenizer.from_pretrained("jackaduma/SecBERT")
global_cyber_model = AutoModelForMaskedLM.from_pretrained("jackaduma/SecBERT")
global_cyber_ner = None


def initialize_hybrid_ner() -> (
    Callable[[List[str]], List[List[Dict[str, Union[str, float]]]]]
):
    def hybrid_ner_batch(texts: List[str]) -> List[List[Dict[str, Union[str, float]]]]:
        global global_cyber_ner, global_cyber_tokenizer, global_cyber_model
        out = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                out.append([])
                continue
            masked, preserved = detect_and_preserve_entities(text)
            new_tokens = [
                ph for ph in preserved if ph not in global_cyber_tokenizer.get_vocab()
            ]
            if new_tokens:
                global_cyber_tokenizer.add_special_tokens(
                    {"additional_special_tokens": new_tokens}
                )
                global_cyber_model.resize_token_embeddings(len(global_cyber_tokenizer))
                global_cyber_ner = None
            if global_cyber_ner is None:
                global_cyber_ner = pipeline(
                    "ner",
                    model=global_cyber_model,
                    tokenizer=global_cyber_tokenizer,
                    aggregation_strategy="simple",
                )
            ents = global_cyber_ner(masked)
            final = []
            for e in ents:
                w = e.get("word")
                if w in preserved:
                    lbl, orig = preserved[w]
                    final.append({"word": orig, "entity": lbl, "score": e.get("score")})
                else:
                    final.append(e)
            out.append(final)
        return out

    return hybrid_ner_batch


# --------------------------
# 4) OpenAI-based NER
# --------------------------
def initialize_openai_ner() -> Callable[[List[str]], List[List[Dict[str, str]]]]:
    """
    Returns a function that takes a list of texts and returns NER results by querying OpenAI GPT.
    Each entity is a dict with keys: "word" and "entity".
    """

    def openai_ner_batch(texts: List[str]) -> List[List[Dict[str, str]]]:
        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append([])
                continue
            prompt = (
                "Extract named entities from the following text. "
                'Return a JSON array of {"word": <entity>, "entity": <label>} entries. '
                f'Text: "{text}"'
            )
            completion = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an NER assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = completion.choices[0].message.content
            try:
                ents = pd.io.json.loads(content)
            except Exception:
                ents = []
            results.append(ents)
        return results

    return openai_ner_batch


# ----------------------------------------------------
# 5) Generic run_ner_on_column using any NER function
# ----------------------------------------------------


def run_ner_on_column(
    df: pd.DataFrame,
    ner_batch_func: Callable[[List[str]], List[List]],
    batch_size: int = 16,
    column: str = TEXT_COL,
) -> pd.DataFrame:
    records = []
    texts = df[column].astype(str).tolist()
    total = len(texts)
    for start in range(0, total, batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_results = ner_batch_func(batch_texts)
        for idx, ents in enumerate(batch_results):
            row_idx = start + idx
            base = {
                col: df.iloc[row_idx][col]
                for col in ["post_id", "comment_id", "source", "labels", "text_orig"]
            }
            keywords = set([e[0] for e in ents])
            keywords.union(
                df.iloc[row_idx].get("keywords", set([]))
            )  # Merge with existing entities if any
            record = {
                **base,
                "text": df.iloc[row_idx][column],
                "keywords": keywords,
            }
            records.append(record)
    return pd.DataFrame(records)


# ----------------------------------------------------
# 6) Extract Entities Using Selected Method
# ----------------------------------------------------


def extract_entities_from_dataframes(
    df,
    method: str = "bert",
    batch_size: int = 16,
    model_name: Optional[str] = None,
) -> pd.DataFrame:
    if method == "bert":
        ner_func = (
            initialize_bert_pipeline(model_name)
            if model_name
            else initialize_bert_pipeline()
        )
    elif method == "stanford":
        ner_func = initialize_stanford_ner()
    elif method == "hybrid":
        ner_func = initialize_hybrid_ner()
    elif method == "openai":
        ner_func = initialize_openai_ner()
    else:
        raise ValueError(f"Unsupported NER method '{method}'")

    df = run_ner_on_column(
        df,
        ner_batch_func=ner_func,
        batch_size=batch_size,
    )
    return df
