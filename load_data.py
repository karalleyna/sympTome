import json
import re
import pandas as pd

from typing import Union, List, Dict, Tuple

POST_ID = "post_id"
COMMENT_ID = "comment_id"
USER = "user"
TITLE_COL = "title"
DESCRIPTION_COL = "description"
TEXT_COL = "text"

DATA_COLUMNS = (POST_ID, USER, TITLE_COL, DESCRIPTION_COL, "comments")
POST_COLUMNS = DATA_COLUMNS[:4]
COMMENT_COLUMNS = (POST_ID, COMMENT_ID, USER, TEXT_COL)


TITLE_CLS = 0
DESCRIPTION_CLS = 1
COMMENT_CLS = 2


def read_json_dataset(file_path: str) -> List[Dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load or parse JSON file '{file_path}': {e}")

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON structure to be a list.")
    return data


def normalize_id(raw_id: Union[str, int]) -> Union[int, str]:
    """Normalizes raw ID by extracting numerical part if applicable."""
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        if raw_id.isdigit():
            return int(raw_id)
        match = re.match(r"^([a-zA-Z]*)(\d+)$", raw_id)
        if match:
            return int(match.group(2))
    return raw_id


def get_invalid_and_duplicate_entries(
    entries: List[Dict], key: str
) -> Tuple[Dict[Union[int, str], List[int]], List[int]]:
    """
    Identifies duplicate or invalid IDs in entries and returns a mapping and list.
    """
    valid_indices = set()
    invalid_indices = []

    for idx, entry in enumerate(entries):
        raw_id = normalize_id(entry.get(key))
        entry[key] = raw_id

        if isinstance(raw_id, int) and raw_id not in valid_indices:
            valid_indices.add(raw_id)
        else:
            invalid_indices.append(idx)

    next_id = max(valid_indices) + 1 if valid_indices else 0
    return invalid_indices, next_id


def validate_ids(entries: List[Dict], key: str) -> None:
    """
    Assigns new unique IDs to entries with duplicate or missing identifiers.
    """
    invalid_indices, next_id = get_invalid_and_duplicate_entries(entries, key)
    for id_val in invalid_indices:
        new_id = f"{next_id:03d}"
        entries[id_val][key] = new_id
        next_id += 1

    return entries


def json_to_dataframes(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    data = read_json_dataset(file_path)

    if not data:
        raise ValueError("The dataset is empty or not properly formatted.")

    # Lists to collect post entries
    posts = []
    # List to collect comments
    comments = []

    for post in data:
        post_id = post.get(POST_ID)
        title_text = post.get(TITLE_COL, "").strip()
        desc_text = post.get(DESCRIPTION_COL, "").strip()

        # Append title row (is_description=0, comment_id=NaN)
        posts.append(
            {
                POST_ID: post_id,
                USER: post.get(USER, ""),
                TEXT_COL: title_text,
                "source": TITLE_CLS,
                COMMENT_ID: pd.NA,
            }
        )

        # Append description row (is_description=1, comment_id=NaN) if description exists
        if desc_text:
            posts.append(
                {
                    POST_ID: post_id,
                    USER: post.get(USER, ""),
                    TEXT_COL: desc_text,
                    "source": DESCRIPTION_CLS,
                    COMMENT_ID: pd.NA,
                }
            )

        # Process comments/treatments
        for comment in post.get("comments", []):
            comments.append(
                {
                    POST_ID: post_id,
                    COMMENT_ID: comment.get(COMMENT_ID, pd.NA),
                    USER: comment.get(USER, ""),
                    TEXT_COL: comment.get(TEXT_COL, "").strip(),
                    "source": COMMENT_CLS,
                }
            )

    # Create DataFrames
    df_posts = pd.DataFrame(posts)
    df_comments = pd.DataFrame(comments)

    return pd.concat([df_posts, df_comments], axis=0, ignore_index=True)
