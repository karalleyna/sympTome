import json
import os
import logging
from typing import Set, List, Dict, Any

import pandas as pd
from tqdm.auto import tqdm
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is set in environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("Please set the OPENAI_API_KEY environment variable.")
openai.api_key = api_key


def classify_text_openai(
    keywords: Set[str], labels: Set[str], text: str
) -> List[Dict[str, Any]]:
    """
    Given a text snippet that may contain multiple parts (symptom, cause, solution),
    ask the OpenAI model to identify each part and return a list of dicts with 'node_type' and 'text'.
    node_type: 0 for symptom, 1 for cause, 2 for solution.
    Returns:
        List[Dict[str, Any]]: e.g. [{"node_type":0,"text":"..."}, ...]
        If classification fails, returns an empty list.
    """
    prompt = f"""
You are an expert at building knowledge graphs for computer and network troubleshooting.
A row of data has already been through NER, giving us:
- text snippet: \"{text}\"
- keywords extracted (as a Python set): {sorted(keywords)}
- labels assigned (as a Python set): {sorted(labels)}

The text snippet may contain multiple parts (e.g., a symptom description and a cause recommendation).
Identify each distinct part and classify it as a symptom, a cause, or a solution.
– For each part, output a JSON object with two keys:
  • "node_type": 0 (symptom), 1 (cause), or 2 (solution)
  • "text": the exact substring from the original snippet corresponding to that part.

Return a JSON array of these objects. Do NOT output any extra commentary.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a background LLM for multi-part classification of network‐troubleshooting text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return []

    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
        # Validate list of dicts
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict) and "node_type" in item and "text" in item:
                    try:
                        node = int(item["node_type"])
                        segment = str(item["text"])
                        results.append({"node_type": node, "text": segment})
                    except Exception:
                        continue
            return results
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse model output as JSON: '{content}'")
    return []


def classify_text(df: pd.DataFrame, method_name: str = "openai") -> pd.DataFrame:
    """
    For each row in DataFrame, classify multi-part text into (node_type, text) pairs,
    then expand DataFrame so each row corresponds to one part.
    Adds 'node_type' and 'segment' columns, dropping the original 'text'.
    Args:
        df (pd.DataFrame): Must contain 'keywords', 'labels', and 'text' columns.
        method_name (str): Classification method. Currently only 'openai' is supported.
    Returns:
        pd.DataFrame: Exploded DataFrame with columns ['keywords', 'labels', 'segment', 'node_type', ...other original columns except 'text']
    """
    supported_methods = {
        "openai": classify_text_openai,
    }
    method = method_name.lower()
    if method not in supported_methods:
        raise ValueError(
            f"Unknown method: {method_name}. Available methods: {list(supported_methods.keys())}"
        )

    classify_fn = supported_methods[method]

    # Prepare for progress apply
    tqdm.pandas(desc="Classifying and splitting rows")

    def _apply_row(row):
        keywords = set(row["keywords"]) if not pd.isna(row["keywords"]) else set()
        labels = set(row["labels"]) if not pd.isna(row["labels"]) else set()
        text = row["text"] if not pd.isna(row["text"]) else ""
        parts = classify_fn(keywords, labels, text)
        # Attach original index or other columns if needed
        return parts

    # Apply to each row, obtaining a list of {node_type, text} for each row
    df["classified_parts"] = df.progress_apply(_apply_row, axis=1)

    # Explode the DataFrame so each part becomes its own row
    exploded = df.explode("classified_parts").reset_index(drop=True)

    # Drop rows with no classification result
    exploded = exploded[exploded["classified_parts"].notna()]

    # Create separate columns for node_type and segment text
    exploded["node_type"] = exploded["classified_parts"].apply(
        lambda x: x.get("node_type")
    )
    exploded["segment"] = exploded["classified_parts"].apply(lambda x: x.get("text"))

    # Drop intermediary and original text columns if desired
    exploded = exploded.drop(columns=["text", "classified_parts"])
    # Optionally reorder columns: put segment and node_type front
    cols = exploded.columns.tolist()
    # Move 'segment' and 'node_type' to front
    new_order = ["segment", "node_type"] + [
        c for c in cols if c not in ["segment", "node_type"]
    ]
    exploded = exploded[new_order]

    return exploded


if __name__ == "__main__":
    # Example usage (illustrative, no code execution needed):
    # Suppose we have one row where the text contains both a symptom and a cause:
    # keywords = ['device', 'error', 'DNS']
    # labels   = ['network', 'troubleshooting']
    # text     = "Device not connecting to network (symptom); possible DNS issue causing failure (cause)."
    # After running classify_text on this single-row DataFrame, the output would be two rows:
    #   segment: "Device not connecting to network"
    #   node_type: 0  (symptom)
    # and
    #   segment: "possible DNS issue causing failure"
    #   node_type: 1  (cause)
    pass
