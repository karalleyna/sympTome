import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from typing import Any, Dict, List, Optional
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns


def plot_clustering_results_2d(
    X: np.ndarray,
    labels: np.ndarray,
    method_name: str = "clustering",
    reducer: str = "pca",
    random_state: int = 0,
    **reducer_kwargs: Any,
) -> None:
    """
    Reduce a high‐dimensional embedding array X to 2D (via PCA, t-SNE, or UMAP)
    and scatter‐plot the points colored by `labels`, using Seaborn.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features)
    - labels: np.ndarray of shape (n_samples,), cluster assignments
    - method_name: str, title prefix for the plot
    - reducer: {'pca', 'tsne', 'umap'}, which 2D reduction to apply
    - random_state: int, seed for PCA/TSNE/UMAP
    - reducer_kwargs: additional keyword args for PCA, TSNE, or UMAP
      (e.g. n_neighbors, min_dist for UMAP; n_components is fixed to 2)

    Returns:
    - None (displays a seaborn-enhanced matplotlib figure)
    """
    reducer_lower = reducer.lower()
    if reducer_lower == "pca":
        model = PCA(n_components=2, random_state=random_state, **reducer_kwargs)
        X_2d = model.fit_transform(X)
        title_suffix = "PCA"
    elif reducer_lower == "tsne":
        model = TSNE(n_components=2, random_state=random_state, **reducer_kwargs)
        X_2d = model.fit_transform(X)
        title_suffix = "t-SNE"
    elif reducer_lower == "umap":
        # UMAP's n_components is always 2 for a 2D projection
        model = UMAP(n_components=2, random_state=random_state, **reducer_kwargs)
        X_2d = model.fit_transform(X)
        title_suffix = "UMAP"
    else:
        raise ValueError("`reducer` must be one of 'pca', 'tsne', or 'umap'.")

    # Prepare a DataFrame for Seaborn
    import pandas as pd

    df_plot = pd.DataFrame({"dim1": X_2d[:, 0], "dim2": X_2d[:, 1], "cluster": labels})

    plt.figure(figsize=(6, 5))
    ax = sns.scatterplot(
        data=df_plot,
        x="dim1",
        y="dim2",
        hue="cluster",
        palette="tab10",
        s=30,
        edgecolor="k",
        linewidth=0.3,
        legend="full",
        alpha=0.8,
    )
    ax.set_title(f"{method_name} (2D projection via {title_suffix})")
    ax.set_xlabel(f"{title_suffix} component 1")
    ax.set_ylabel(f"{title_suffix} component 2")
    ax.grid(alpha=0.3, linestyle="--")
    plt.legend(title="Cluster Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_clustering_metrics_barchart(
    X: np.ndarray, labels: np.ndarray, title: str = "Clustering Metrics"
) -> None:
    """
    Compute silhouette, Calinski-Harabasz, and Davies-Bouldin scores for a given labeling,
    then display a bar chart of those three metrics.

    Parameters:
    - X: np.ndarray of shape (n_samples, n_features)
    - labels: np.ndarray of shape (n_samples,), cluster assignments
              (labels == –1 will be treated as “noise” and excluded from metric computations)
    - title: str, title for the bar‐chart

    Returns:
    - None (displays a matplotlib bar chart)
    """
    mask = labels != -1
    unique_labels = set(labels[mask])

    metrics: Dict[str, Optional[float]] = {
        "Silhouette": None,
        "Calinski-Harabasz": None,
        "Davies-Bouldin": None,
    }

    if len(unique_labels) > 1:
        metrics["Silhouette"] = silhouette_score(X[mask], labels[mask])
        metrics["Calinski-Harabasz"] = calinski_harabasz_score(X[mask], labels[mask])
        metrics["Davies-Bouldin"] = davies_bouldin_score(X[mask], labels[mask])

    # Prepare bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(metrics.keys())
    values = [metrics[n] if metrics[n] is not None else 0.0 for n in names]
    bars = ax.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title(title)
    ax.set_ylim(0, max(values) * 1.1 if max(values) > 0 else 1)

    # Annotate bars with numeric values or “N/A”
    for bar, name in zip(bars, names):
        height = bar.get_height()
        label = f"{metrics[name]:.3f}" if metrics[name] is not None else "N/A"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * (max(values) if max(values) > 0 else 1),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_kmeans_metrics_over_k(
    X: np.ndarray, k_range: List[int], random_state: Optional[int] = 0
) -> None:
    """
    Run K-Means for each k in k_range, compute silhouette, Calinski-Harabasz,
    and Davies-Bouldin scores, and plot each metric as a function of k.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
    - k_range: List[int], list of cluster‐counts to evaluate (e.g. [2, 3, 4, 5, 6, 7, 8])
    - random_state: int or None, seed passed to KMeans for reproducibility

    Returns:
    - None (displays a matplotlib figure with three curves)
    """
    from sklearn.cluster import KMeans

    silhouette_vals: List[float] = []
    ch_vals: List[float] = []
    db_vals: List[float] = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X)

        # If only one cluster (which shouldn’t happen for k≥2), skip metrics
        if len(set(labels)) > 1:
            mask = labels != -1  # KMeans never produces –1, but keep consistent
            silhouette_vals.append(silhouette_score(X[mask], labels[mask]))
            ch_vals.append(calinski_harabasz_score(X[mask], labels[mask]))
            db_vals.append(davies_bouldin_score(X[mask], labels[mask]))
        else:
            silhouette_vals.append(np.nan)
            ch_vals.append(np.nan)
            db_vals.append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_range, silhouette_vals, marker="o", label="Silhouette")
    ax.plot(k_range, ch_vals, marker="s", label="Calinski-Harabasz")
    ax.plot(k_range, db_vals, marker="^", label="Davies-Bouldin")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_xticks(k_range)
    ax.set_title("K-Means Evaluation Metrics over k")
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_er_diagram():
    import graphviz

    dot = graphviz.Digraph(comment="Support Case ERD", format="png")
    dot.attr(dpi="300")

    # Define nodes with uppercase labels, lowercase types, and colors
    dot.node(
        "P",
        """POST\n- postId: string\n- username: string\n- problemDescription: string\n- rawReplies: list<string>""",
        shape="box",
        style="filled",
        fillcolor="lightblue",
    )

    dot.node(
        "S",
        """SYMPTOM\n- symptomId: string\n- name: string\n- description: string\n- category: string""",
        shape="box",
        style="filled",
        fillcolor="lightyellow",
    )

    dot.node(
        "C",
        """CAUSE\n- causeId: string\n- name: string\n- description: string\n- category: string""",
        shape="box",
        style="filled",
        fillcolor="lightcoral",
    )

    dot.node(
        "R",
        """SOLUTION\n- solutionId: string\n- name: string\n- description: string\n- category: string""",
        shape="box",
        style="filled",
        fillcolor="lightgreen",
    )

    # Define edges with uppercase labels
    dot.edge("P", "S", label="RELATES_TO")
    dot.edge("S", "C", label="EXPLAINED_BY")
    dot.edge("S", "R", label="RESOLVED_BY")

    # Render and save the image
    dot.render("support_case_erd_final", format="png", cleanup=True)
