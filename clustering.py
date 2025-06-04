import numpy as np
from typing import Any, Dict, Optional, Tuple

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from hdbscan import HDBSCAN


def run_kmeans_clustering(
    X: np.ndarray,
    n_clusters: int = 5,
    init: str = "k-means++",
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, KMeans]:

    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def run_agglomerative_clustering(
    X: np.ndarray,
    n_clusters: int = 8,
    affinity: str = "euclidean",
    linkage: str = "ward",
) -> Tuple[np.ndarray, AgglomerativeClustering]:
    agg = AgglomerativeClustering(
        n_clusters=n_clusters, affinity=affinity, linkage=linkage
    )
    labels = agg.fit_predict(X)
    return labels, agg


def run_dbscan_clustering(
    X: np.ndarray, eps: float = 0.5, min_samples: int = 5, metric: str = "euclidean"
) -> Tuple[np.ndarray, DBSCAN]:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = dbscan.fit_predict(X)
    return labels, dbscan


def run_hdbscan_clustering(
    X: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: Optional[int] = 2,
    metric: str = "euclidean",
    method: str = "eom",
) -> Tuple[np.ndarray, HDBSCAN]:
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric=metric,
        cluster_selection_method=method,
    ).fit(X)
    return hdbscan_model.labels_, hdbscan_model


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray
) -> Dict[str, Optional[float]]:

    mask = labels != -1
    scores: Dict[str, Optional[float]] = {}

    unique_labels = set(labels[mask])
    if len(unique_labels) > 1:
        scores["silhouette"] = silhouette_score(X[mask], labels[mask])
        scores["calinski_harabasz"] = calinski_harabasz_score(X[mask], labels[mask])
        scores["davies_bouldin"] = davies_bouldin_score(X[mask], labels[mask])
    else:
        scores["silhouette"] = None
        scores["calinski_harabasz"] = None
        scores["davies_bouldin"] = None

    return scores


def cluster(
    embeddings: np.ndarray,
    method="kmeans",
    params: Optional[Dict[str, Dict[str, Any]]] = {},
    evaluate: bool = True,
) -> Dict[str, Dict[str, Any]]:
    results = {}

    if method == "kmeans":
        labels, model = run_kmeans_clustering(embeddings, **params)

    elif method == "agglomerative":
        labels, model = run_agglomerative_clustering(embeddings, **params)

    elif method == "dbscan":
        labels, model = run_dbscan_clustering(embeddings, **params)

    elif method == "hdbscan":
        labels, model = run_hdbscan_clustering(embeddings, **params)

    else:
        raise ValueError(
            f"Unsupported clustering method '{method}'. "
            f"Choose from ['kmeans', 'agglomerative', 'dbscan', 'hdbscan']."
        )

    result_entry: Dict[str, Any] = {"labels": labels, "model": model}

    if evaluate:
        result_entry["scores"] = evaluate_clustering(embeddings, labels)

    results[method] = result_entry

    return results
