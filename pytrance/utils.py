from typing import Any, List, Sequence, Tuple

import numpy as np
from anndata import AnnData
from pandas import DataFrame
from scipy.sparse import csr_matrix, spmatrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from torch import Size, from_numpy, sparse_coo_tensor


def get_neighbors(
    cell_df: DataFrame,
    radius: int = 2,
    n_neighbors: int = 20,
    metric: str = "euclidean",
    mode: str = "connectivity",
    x_key="x_pos",
    y_key="y_pos",
    z_key=None,
    n_jobs: int = 1,
    include_self: bool = False,
) -> tuple[csr_matrix, csr_matrix]:
    """Build a spatial neighbor graph from transcript coordinates.

    Parameters
    ----------
    cell_df : DataFrame
        DataFrame containing transcript information of input cell.
    radius : int, optional
        Search radius for neighbors. If provided, radius_neighbors_graph is used.
        Default is 2.
    n_neighbors : int, optional
        Number of nearest neighbors to consider when radius is None.
        Default is 20.
    metric : str, optional
        Distance metric to use. Default is "euclidean".
    mode : str, optional
        Type of graph returned. "connectivity" returns binary adjacency matrix,
        "distance" returns distance values. Default is "connectivity".
    x_key : str, optional
        Column name for x coordinates in cell_df. Default is "x_pos".
    y_key : str, optional
        Column name for y coordinates in cell_df. Default is "y_pos".
    z_key : str, optional
        Column name for z coordinates in cell_df. If None, 2D coordinates are used.
        Default is None.
    n_jobs : int, optional
        Number of parallel jobs for neighbor computation. Default is 1.
    include_self : bool, optional
        Whether to include self-loops in the graph. Default is False.

    Returns
    -------
    tuple[csr_matrix, csr_matrix]
        Sparse adjacency matrix in CSR format representing the neighbor graph.
    """

    if z_key:
        coords = np.dstack(
            (cell_df[x_key].values, cell_df[y_key].values, cell_df[z_key])
        ).squeeze()
    else:
        coords = np.dstack((cell_df[x_key].values, cell_df[y_key].values)).squeeze()

    if radius is not None:  # build graph based on radius/distance
        graph = radius_neighbors_graph(
            coords,
            radius=radius,
            mode=mode,
            metric=metric,
            n_jobs=n_jobs,
            include_self=include_self,
        )
    else:  # build graph based on nearest neighbors
        graph = kneighbors_graph(
            coords,
            n_neighbors=n_neighbors,
            metric=metric,
            mode=mode,
            n_jobs=n_jobs,
            include_self=include_self,
        )

    return graph


def get_gene_subclusters(genes: Sequence[Any], clustering_model: Any) -> dict:
    """Group genes by their assigned subclusters from a clustering model.

    Parameters
    ----------
    genes : array-like
        Array of gene names or identifiers.
    clustering_model : object
        Fitted clustering model with `labels_` attribute containing cluster assignments.

    Returns
    -------
    dict
        Dictionary mapping subcluster IDs to lists of genes in that subcluster.
        Keys are unique cluster labels, values are lists of gene names.
    """
    cluster_gene_groups_subset = {}
    for subcluster in np.unique(clustering_model.labels_):
        cluster_idxs = np.where(clustering_model.labels_ == subcluster)[0]
        cluster_genes = np.array(genes)[cluster_idxs]
        cluster_gene_groups_subset[int(subcluster)] = cluster_genes.tolist()
        
    return cluster_gene_groups_subset


# from DGI
def sparse_mx_to_torch_sparse_tensor(sparse_mx: spmatrix):
    """Convert a scipy sparse matrix to a PyTorch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse matrix
        Input sparse matrix (converted to COO format internally).

    Returns
    -------
    torch.sparse.FloatTensor
        Sparse tensor in COO format with float32 dtype.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = from_numpy(sparse_mx.data)
    shape = Size(sparse_mx.shape)
    return sparse_coo_tensor(indices, values, shape)


def get_tree_linkage(model: Any) -> np.ndarray:
    """Generate a linkage matrix from a hierarchical clustering model.

    Parameters
    ----------
    model : object
        Fitted hierarchical clustering model with `children_`, `distances_`,
        and `labels_` attributes.

    Returns
    -------
    ndarray
        Linkage matrix
    """
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix


def compute_figure_size(
    adata: AnnData,
    cells: Sequence[Any],
    n_rows: int,
    n_cols: int,
    scale: float = 0.1,
    col_padding: float = 1,
    row_padding: float = 20,
    boundary_padding: float = 1,
    cell_key: str = "cell",
    x_key: str = "x",
    y_key: str = "y",
) -> Tuple[
    float,
    float,
    List[float],
    List[float],
    List[List[float]],
    List[List[float]],
    float,
    float,
]:
    """Calculate optimized figure dimensions for multi-panel subplot layouts.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing observation metadata and coordinates.
    cells : array-like
        List of cell identifiers to include in the layout, ordered by subplot position.
    n_rows : int
        Number of rows in the subplot grid.
    n_cols : int
        Number of columns in the subplot grid.
    scale : float, optional
        Scale factor applied to subplot dimensions. Default is 0.1.
    col_padding : float, optional
        Horizontal padding between columns. Default is 1.
    row_padding : float, optional
        Vertical padding between rows. Default is 20.
    boundary_padding : float or int, optional
        Padding around data boundaries. If < 1, treated as relative (fraction of range);
        if >= 1, treated as absolute units. Default is 1.
    cell_key : str, optional
        Column name in adata.obs for cell grouping. Default is "cell".
    x_key : str, optional
        Column name in adata.obs for x coordinates. Default is "x".
    y_key : str, optional
        Column name in adata.obs for y coordinates. Default is "y".

    Returns
    -------
    tuple
        A tuple of 8 elements:
        - fig_width : float - Final figure width in inches
        - fig_height : float - Final figure height in inches
        - row_heights : list[float] - Height of each row
        - col_widths : list[float] - Width of each column
        - subplot_heights : list[list[float]] - Height of each subplot
        - subplot_widths : list[list[float]] - Width of each subplot
        - col_padding : float - Applied column padding
        - row_padding : float - Applied row padding
    """
    # max figure size scales with the grid dimensions (cols -> width, rows -> height)
    max_fig_width = n_cols * 5
    max_fig_height = n_rows * 5

    x_ranges = []
    y_ranges = []
    adata_cell_grouped = adata.obs.groupby(cell_key, observed=False)
    cell = 0
    for r in range(n_rows):
        x_ranges.append([])
        y_ranges.append([])
        for c in range(n_cols):
            cell_df = adata_cell_grouped.get_group(cells[cell])
            x_ranges[r].append((cell_df[x_key].min(), cell_df[x_key].max()))
            y_ranges[r].append((cell_df[y_key].min(), cell_df[y_key].max()))
            cell += 1

    # Step 1: Calculate natural subplot sizes (scaled by `scale`), accounting for boundary padding
    # boundary_padding can be absolute (>=1) or relative (<1)
    subplot_widths = []
    subplot_heights = []
    for r, row in enumerate(x_ranges):
        subplot_widths.append([])
        subplot_heights.append([])
        for c, (xmin, xmax) in enumerate(row):
            x_range = xmax - xmin
            y_range = y_ranges[r][c][1] - y_ranges[r][c][0]
            # Apply boundary padding to ranges
            if boundary_padding < 1:
                x_pad = x_range * boundary_padding
                y_pad = y_range * boundary_padding
            else:
                x_pad = boundary_padding
                y_pad = boundary_padding
            # Calculate subplot size with padding included
            subplot_widths[r].append((x_range + 2 * x_pad) * scale)
            subplot_heights[r].append((y_range + 2 * y_pad) * scale)

    row_heights = [max(row) for row in subplot_heights]
    col_widths = [max(col) for col in zip(*subplot_widths)]

    # Step 2: Compute natural figure size including padding
    natural_fig_width = sum(col_widths) + col_padding * (n_cols - 1)
    natural_fig_height = sum(row_heights) + row_padding * (n_rows - 1)

    # Step 3: Compute downscale factor
    width_scale_factor = (
        max_fig_width / natural_fig_width if natural_fig_width > max_fig_width else 1.0
    )
    height_scale_factor = (
        max_fig_height / natural_fig_height
        if natural_fig_height > max_fig_height
        else 1.0
    )

    # Choose the most constraining factor
    downscale = min(width_scale_factor, height_scale_factor)

    # Step 3: Apply downscale to subplot sizes and padding
    subplot_widths = [[w * downscale for w in row] for row in subplot_widths]
    subplot_heights = [[h * downscale for h in row] for row in subplot_heights]

    row_heights = [max(row) for row in subplot_heights]
    row_padding *= downscale
    col_padding *= downscale

    # keep per-column width for backwards compatibility
    col_widths = [max(col) for col in zip(*subplot_widths)]

    # Figure size is based on the widest row (since columns can vary per row)
    row_total_widths = [sum(row) + col_padding * (n_cols - 1) for row in subplot_widths]
    fig_width = max(row_total_widths)
    fig_height = sum(row_heights) + row_padding * (n_rows - 1)

    return (
        fig_width,
        fig_height,
        row_heights,
        col_widths,
        subplot_heights,
        subplot_widths,
        col_padding,
        row_padding,
    )
