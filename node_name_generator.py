import os
import pandas as pd
import openai
import numpy as np
from embedding import embed
from clustering import cluster

# Set your API key
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_node_names(
    df,
    embedding_type="openai",
    suffix="name",
    output_path="/tmp/embeddings.csv",
    clustering_method="hdbscan",
    clustering_params={"min_cluster_size": 5},
):
    df = df.copy()
    text_column = "text_" + suffix
    df[text_column] = (
        df["segment"].astype(str)
        + " "
        + df["labels"].astype(str)
        + " "
        + df["keywords"].astype(str)
    )

    embeddings_array = embed(
        df,
        embedding_type=embedding_type,
        output_path=output_path,
        text_column=text_column,
    )
    df["embedding"] = embeddings_array.tolist()
    df["node_name"] = None

    for node_type, group in df.groupby("node_type"):
        indices = group.index.tolist()
        embeddings = np.array(group["embedding"].tolist())

        clustering_results = cluster(
            embeddings, method=clustering_method, params=clustering_params
        )
        labels = clustering_results[clustering_method]["labels"]
        probs = clustering_results[clustering_method]["model"].probabilities_

        df.loc[indices, "cluster_label"] = labels

        label_to_indices = {}
        for i, label in enumerate(labels):
            label_to_indices.setdefault(label, []).append(i)

        for label, local_inds in label_to_indices.items():
            global_inds = [indices[i] for i in local_inds]
            if label == -1:
                for i in global_inds:
                    row = df.loc[i]
                    prompt = f"Segment: {row['segment']}, Label: {row['labels']}, Keywords: {row['keywords']}\nGenerate a short descriptive name."
                    name = query_llm(prompt)
                    df.loc[i, "node_name"] = name
            else:
                rep_inds = sorted(local_inds, key=lambda i: probs[i], reverse=True)[:3]
                reps = group.iloc[rep_inds]
                rep_prompt = "\n".join(
                    f"Segment: {r['segment']}, Label: {r['labels']}, Keywords: {r['keywords']}"
                    for _, r in reps.iterrows()
                )
                prompt = (
                    f"Cluster items:\n{rep_prompt}\nGenerate a short descriptive name."
                )
                name = query_llm(prompt)
                for i in global_inds:
                    df.loc[i, "node_name"] = name

    df.drop(columns=[text_column, "cluster_label"], inplace=True, errors="ignore")
    df = refine_node_names(df, name_column="node_name", new_column="refined_node_name")

    df.drop(columns=["node_name"], inplace=True)
    df.drop(columns=["segment"], inplace=True)
    # Change refined_node_name to node_name
    df.rename(columns={"refined_node_name": "node_name"}, inplace=True)
    # Drop keywords embedding and labels
    df.drop(columns=["keywords", "embedding", "labels"], inplace=True)

    return df


def query_llm(prompt):
    # Updated to use the new OpenAI Python 1.x interface
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates short descriptive names.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=10,
        n=1,
    )
    # The response structure remains the same for accessing the generated text
    return response.choices[0].message.content.strip()


import openai
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def refine_node_names(df, name_column="node_name", new_column="refined_node_name"):
    """
    Refines node names to short, snake_case descriptive tokens like 'wifi_connectivity'.
    Uses OpenAI to transform each unique name.
    """
    df = df.copy()
    unique_names = df[name_column].dropna().unique()
    name_map = {}

    for name in unique_names:
        prompt = (
            "You are a helpful assistant. Convert the phrase below into a concise lowercase snake_case noun phrase tag.\n"
            "Example: 'WiFi Connectivity Troubleshooting Kit' -> 'wifi_connectivity'\n"
            f"Phrase: '{name}'\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You convert verbose titles into compact lowercase snake_case noun phrase labels.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
            n=1,
        )
        refined = response.choices[0].message.content.strip()
        name_map[name] = refined

    df[new_column] = df[name_column].map(name_map)
    return df
