from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from anndata import AnnData
from pandas import DataFrame
from scanpy.pp import neighbors
from scanpy.tl import leiden
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

from .plotting import dendrogram
from .utils import get_gene_subclusters


def aggregate_transcript_embeddings(
    node_embeds: np.ndarray,
    adata: AnnData,
    gene_key: str = "gene",
) -> DataFrame:
    """Aggregate transcript embeddings by gene using mean pooling.

    Parameters
    ----------
    node_embeds : ndarray
        Transcript embedding matrix.
    adata : AnnData
        Annotated data object containing observation metadata.
    gene_key : str, optional
        Column name in adata.obs for gene assignment. Default is "gene".

    Returns
    -------
    DataFrame
        Gene-level embeddings with genes as index and embedding dimensions as columns.
    """

    if node_embeds.shape[0] != adata.n_obs:
        raise ValueError(
            f"node_embeds and adata have incompatible shapes: "
            f"node_embeds.shape[0]={node_embeds.shape[0]} != adata.n_obs={adata.n_obs}"
        )
    
    embeds_aggregated = DataFrame(
        columns=np.arange(node_embeds.shape[1]), index=adata.var_names
    )
    adata_obs_grouped = adata.obs.groupby(gene_key, observed=True)
    for gene in tqdm(adata.var_names):
        transcript_ids = adata_obs_grouped.get_group(gene).id.values
        emb_aggr = np.mean(node_embeds[transcript_ids], axis=0)
        embeds_aggregated.loc[gene] = emb_aggr

    return embeds_aggregated


def cluster_gene_embeddings(
    embeds: np.ndarray,
    n_clusters: int = 2,
    adata: Optional[AnnData] = None,
    algo: str = "kmeans",
    return_labels: bool = False,
    seed: int = 808,
) -> Optional[np.ndarray]:
    """Cluster gene embeddings using k-means or agglomerative clustering.

    Parameters
    ----------
    embeds : ndarray
        Gene embedding matrix.
    n_clusters : int, optional
        Number of clusters. Default is 2.
    adata : AnnData, optional
        Annotated data object containing observation metadata. Default is None.
    algo : {'kmeans', 'agglomerative'}, optional
        Clustering algorithm. Default is "kmeans".
    return_labels : bool, optional
        If True, return cluster labels. If False, return None. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 808.

    Returns
    -------
    ndarray or None
        Cluster labels if return_labels is True, otherwise None.
    """

    if algo == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=seed)
    elif algo == "agglomerative":
        model = AgglomerativeClustering(
            metric="manhattan",
            linkage="average",
            n_clusters=n_clusters,
            compute_distances=True,
        )
    else:
        raise ValueError(f"Clustering algorithm '{algo}' not implemented")

    model = model.fit(embeds)
    labels = model.labels_

    # add cluster labels to adata object
    if adata is not None:
        adata.var[algo] = None
        for label in np.unique(labels):
            adata.var.loc[adata.var.index[np.where(labels == label)[0]], algo] = label
        adata.var[algo] = adata.var[algo].astype("category")

    if return_labels:
        return labels
    else:
        return


# special case because it's based on scanpy implementation
def cluster_gene_embeddings_leiden(
    embeds: np.ndarray,
    resolution: float = 1,
    n_neighbors: int = 5,
    adata: Optional[AnnData] = None,
    return_labels: bool = False,
    key_added: str = "leiden",
    flavor: str = 'igraph',
    seed: int = 808,
) -> Optional[np.ndarray]:
    """Cluster gene embeddings using Leiden algorithm.

    Parameters
    ----------
    embeds : ndarray
        Gene embedding matrix.
    resolution : float, optional
        Resolution parameter for Leiden algorithm controlling cluster granularity.
        Default is 1.
    n_neighbors : int, optional
        Number of neighbors for k-nearest neighbor graph construction. Default is 5.
    adata : AnnData, optional
        Annotated data object containing observation metadata. Default is None.
    return_labels : bool, optional
        If True, return cluster labels. If False, return None. Default is False.
    key_added : str, optional
        Column name for cluster labels in adata.var. Default is "leiden".
    flavor : str, Optional
        Flavor to use for leiden clustering, Default is "igraph".
    seed : int, optional
        Random seed for reproducibility. Default is 808.

    Returns
    -------
    array-like or None
        Cluster labels if return_labels is True, otherwise None.
    """

    embeds_adata = AnnData(embeds)
    neighbors(embeds_adata, n_neighbors=n_neighbors)
    leiden(embeds_adata, resolution=resolution, key_added=key_added, flavor=flavor, random_state=seed)

    if adata is not None:
        adata.var[key_added] = embeds_adata.obs[key_added].astype("int")

    print(f"detected {adata.var[key_added].nunique()} clusters")

    if return_labels:
        return embeds_adata.obs[key_added]
    else:
        return


def embedding_pca(
    embeds: DataFrame,
    n_components: int = 2,
    seed: int = 808,
) -> Tuple[np.ndarray, PCA]:
    """Perform PCA dimensionality reduction on gene embeddings.

    Parameters
    ----------
    embeds : DataFrame
        Gene embedding matrix.
    n_components : int, optional
        Number of principal components to compute. Default is 2.
    seed : int, optional
        Random seed for reproducibility. Default is 808.

    Returns
    -------
    tuple
        - embeds_pca : ndarray
            Transformed embeddings of shape (n_genes, n_components).
        - pca : PCA
            Fitted PCA model object.
    """
    pca = PCA(n_components=n_components, random_state=seed)
    embeds_pca = pca.fit_transform(embeds.iloc[:, : embeds.shape[1]])
    return embeds_pca, pca


def subcluster(
    corr_array: np.ndarray,
    genes: Sequence[Any],
    distance_threshold: float = 0.2,
    n_subclusters: Optional[int] = None,
    plot_tree: bool = True,
    gene_names_ordered: Optional[Sequence[Any]] = None,
    metric: str = "euclidean",
    linkage: str = "ward",
) -> Dict:
    """Perform hierarchical clustering on gene embedding correlations.

    Parameters
    ----------
    corr_array : ndarray
        Gene embedding correlation matrix.
    genes : array-like
        Gene names or identifiers corresponding to rows in corr_array.
    distance_threshold : float, optional
        Distance threshold for cluster merging in hierarchical clustering.
        Default is 0.2.
    n_subclusters : int, optional
        If specified, overrides distance_threshold and sets exact number of clusters.
        Default is None.
    plot_tree : bool, optional
        If True, generate and display dendrogram visualization. Default is True.
    gene_names_ordered : array-like, optional
        Ordered gene names for dendrogram visualization. Default is None.
    metric : str, optional
        Distance metric for clustering. Default is "euclidean".
    linkage : str, optional
        Linkage criterion for agglomerative clustering. Default is "ward".

    Returns
    -------
    dict
        Dictionary mapping subcluster IDs to lists of genes in each subcluster.
    """

    clustering_model = AgglomerativeClustering(
        metric=metric,
        linkage=linkage,
        distance_threshold=distance_threshold,
        n_clusters=n_subclusters,
        compute_distances=True,
    )
    clustering_model = clustering_model.fit(corr_array)
    if plot_tree:
        dendro = dendrogram(
            clustering_model,
            color_threshold=distance_threshold,
            gene_names_ordered=gene_names_ordered,
        )

    cluster_gene_sets_subset = get_gene_subclusters(
        genes=genes, clustering_model=clustering_model
    )
    return cluster_gene_sets_subset
