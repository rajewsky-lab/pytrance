from itertools import combinations
from multiprocessing import Manager, Process
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from anndata import AnnData
from pandas import DataFrame
from scipy.sparse import spmatrix
from tqdm import tqdm

from .utils import get_neighbors


def clq_pairwise(
    adata: AnnData,
    genes: Sequence[Any],
    graph: Optional[spmatrix] = None,
    radius: float = 2,
    n_neighbors: Optional[int] = None,
    n_permutations: int = 0,
    cell_key: str = "cell",
    cat_key: str = "gene",
    min_counts: int = 2,
    n_processes: int = 1,
    seed: int = 808,
    **kwargs: Any,
) -> Tuple[Dict, Dict]:
    """Compute co-localization quotient (CLQ) scores between gene pairs.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with transcript-level observations.
    genes : array-like
        List of gene names to compute pairwise CLQ for.
    graph : sparse matrix, optional
        Precomputed spatial neighbor graph. If None, computed from coordinates.
        Default is None.
    radius : float, optional
        Spatial radius for neighbor graph construction. Default is 2.
    n_neighbors : int, optional
        Number of nearest neighbors if radius is None. Default is None.
    n_permutations : int, optional
        Number of permutations for score adjustment. Default is 0.
    cell_key : str, optional
        Column name in adata.obs for cell IDs. Default is "cell".
    cat_key : str, optional
        Column name in adata.obs for gene/category assignment. Default is "gene".
    min_counts : int, optional
        Minimum transcript count per gene per cell to include in analysis.
        Default is 2.
    n_processes : int, optional
        Number of parallel processes for computation. Default is 1.
    seed : int, optional
        Random seed for reproducibility. Default is 808.
    **kwargs
        Additional keyword arguments passed to clq().

    Returns
    -------
    tuple
        - pairwise_clqs : dict
            Dictionary mapping gene pairs to dictionaries containing 'clqs', 'clqs_adj',
            and 'clqs_perm' (raw, adjusted, and permuted CLQ scores by cell).
        - aggregated_norm_clqs : dict
            Aggregated statistics with 'mean' and 'median' keys, mapping gene pairs
            to aggregated normalized CLQ scores.
    """

    gene_pairs = list(combinations(genes, 2))
    if radius is None:  # otherwise clq is symmetric
        gene_pairs.extend(
            list(combinations(genes[::-1], 2))
        )  # add pairs in reverse order
    gene_pairs.extend([(g, g) for g in genes])  # add self pairs
    pairwise_clqs = {}
    for gene_pair in tqdm(gene_pairs, desc="gene pairs"):
        # only keep cells with certain number of transcripts for both genes
        adata_gene_pair = adata.copy()
        for gene in gene_pair:
            gene_cell_counts = (
                adata_gene_pair[adata_gene_pair.obs[cat_key] == gene]
                .obs[cell_key]
                .value_counts()
            )
            gene_cell_counts_idxs = gene_cell_counts.index[
                gene_cell_counts >= min_counts
            ]
            adata_gene_pair = adata_gene_pair[
                adata_gene_pair.obs[cell_key].isin(gene_cell_counts_idxs)
            ]

        cell_clqs, cell_clqs_adjusted, cell_clqs_permuted = clq(
            adata_gene_pair,
            genes=gene_pair,
            graph=graph,
            n_neighbors=n_neighbors,
            radius=radius,
            n_permutations=n_permutations,
            cell_key=cell_key,
            cat_key=cat_key,
            seed=seed,
            pairwise=True,
            verbose=0,
            n_processes=n_processes,
            **kwargs,
        )
        pairwise_clqs[gene_pair] = {
            "clqs": cell_clqs,
            "clqs_adj": cell_clqs_adjusted,
            "clqs_perm": cell_clqs_permuted,
        }

    # for each pair aggregate (mean or median) normalized clq scores
    aggregated_norm_clqs = {"mean": {}, "median": {}}
    for pair, clq_dicts in pairwise_clqs.items():
        clqs_adj_vals = list(clq_dicts["clqs_adj"].values())
        aggregated_norm_clqs["median"][pair] = np.median(clqs_adj_vals)
        aggregated_norm_clqs["mean"][pair] = np.mean(clqs_adj_vals)
        if radius is not None:  # symmetric
            aggregated_norm_clqs["median"][(pair[1], pair[0])] = aggregated_norm_clqs[
                "median"
            ][pair]
            aggregated_norm_clqs["mean"][(pair[1], pair[0])] = aggregated_norm_clqs[
                "mean"
            ][pair]

    return pairwise_clqs, aggregated_norm_clqs


def clq(
    adata: AnnData,
    genes: Union[str, Sequence[Any]],
    graph: Optional[spmatrix] = None,
    radius: float = 2,
    n_neighbors: Optional[int] = None,
    n_permutations: int = 0,
    n_processes: int = 1,
    cell_key: str = "cell",
    cat_key: str = "gene",
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    verbose: int = 1,
    seed: int = 808,
    **kwargs: Any,
) -> Union[Dict, Tuple[Dict, Dict, Dict]]:
    """Compute co-localization quotient (CLQ) scores across all cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with transcript-level observations.
    genes : str or array-like
        Single gene name (str) for self co-localization or list of genes for
        category definitions.
    graph : sparse matrix, optional
        Precomputed spatial neighbor graph. If None, computed from coordinates.
        Default is None.
    radius : float, optional
        Spatial radius for neighbor graph construction. Default is 2.
    n_neighbors : int, optional
        Number of nearest neighbors if radius is None. Default is None.
    n_permutations : int, optional
        Number of permutations for score adjustment. Default is 0.
    n_processes : int, optional
        Number of parallel processes. Default is 1.
    cell_key : str, optional
        Column name in adata.obs for cell IDs. Default is "cell".
    cat_key : str, optional
        Column name in adata.obs for gene/category assignment. Default is "gene".
    x_key : str, optional
        Column name in adata.obs for x coordinates. Default is "x".
    y_key : str, optional
        Column name in adata.obs for y coordinates. Default is "y".
    z_key : str, optional
        Column name in adata.obs for z coordinates. Default is "z".
    verbose : int, optional
        Verbosity level (0=silent, 1+=with progress bars). Default is 1.
    seed : int, optional
        Random seed for permutation testing. Default is 808.
    **kwargs
        Additional keyword arguments passed to clq_single_cell().

    Returns
    -------
    dict or tuple
        If n_permutations > 0, returns tuple of:
        - cell_clqs : dict - Raw CLQ scores per cell
        - cell_clqs_adjusted : dict - Permutation-adjusted CLQ scores per cell
        - cell_clqs_permuted : dict - Permutation distributions per cell

        If n_permutations == 0, returns:
        - cell_clqs : dict - Raw CLQ scores per cell
    """

    if radius is None and n_neighbors is None:
        raise ValueError("either radius or n_neighbors must be specified")

    if cat_key not in adata.obs.columns:
        raise ValueError(f'"{cat_key}" not found in adata.obs')

    if x_key not in adata.obs.columns:
        raise ValueError(f'"{x_key}" not found in adata.obs')

    if y_key not in adata.obs.columns:
        raise ValueError(f'"{y_key}" not found in adata.obs')

    if z_key and z_key not in adata.obs.columns:
        raise ValueError(f'"{z_key}" not found in adata.obs')

    categories = genes
    if type(categories) is str:  # single gene passed as string -> self co-localization
        categories = [categories]
    cat_a = categories
    cat_b = categories

    if n_permutations:
        cell_clqs_permuted = {}

    cell_ids = adata.obs[cell_key].unique()

    # helper that processes a chunk of cells
    def _worker(cells, shared_clqs, shared_norm, shared_perms, adata_sub):
        # local copy of parameters to avoid closure issues
        n_perm = n_permutations
        grouped = adata_sub.obs.groupby(cell_key, observed=False)
        for cell in tqdm(cells, desc="cells", disable=not bool(verbose)):
            cell_df = grouped.get_group(cell)
            if n_perm:
                clq_val, clq_norm, clq_perms = clq_single_cell(
                    cell_df,
                    cat_a,
                    cat_b,
                    cat_key,
                    graph=graph,
                    radius=radius,
                    n_neighbors=n_neighbors,
                    n_permutations=n_perm,
                    x_key=x_key,
                    y_key=y_key,
                    z_key=z_key,
                    seed=seed,
                    **kwargs,
                )
                shared_norm[cell] = clq_norm
                shared_perms[cell] = clq_perms
            else:  # no permutations case
                clq_val = clq_single_cell(
                    cell_df,
                    cat_a,
                    cat_b,
                    cat_key,
                    graph=graph,
                    radius=radius,
                    n_neighbors=n_neighbors,
                    x_key=x_key,
                    y_key=y_key,
                    z_key=z_key,
                    seed=seed,
                    **kwargs,
                )
            shared_clqs[cell] = clq_val

    # split list into n_processes chunks
    cell_ids_split = [
        cell_ids[i : i + max(1, len(cell_ids) // n_processes)]
        for i in range(0, len(cell_ids), max(1, len(cell_ids) // n_processes))
    ]

    # prepare shared dicts
    with Manager() as manager:
        # manager = Manager()
        cell_clqs = manager.dict()
        cell_clqs_adjusted = manager.dict()
        cell_clqs_permuted = manager.dict()

        jobs = []
        for sublist in cell_ids_split:
            adata_sub = adata[adata.obs[cell_key].isin(sublist)]
            if n_processes > 1:
                p = Process(
                    target=_worker,
                    args=(
                        sublist,
                        cell_clqs,
                        cell_clqs_adjusted,
                        cell_clqs_permuted,
                        adata_sub,
                    ),
                )
                p.daemon = True
                jobs.append(p)
                p.start()
            else:
                # run in current process when only one worker
                _worker(
                    sublist,
                    cell_clqs,
                    cell_clqs_adjusted,
                    cell_clqs_permuted,
                    adata_sub,
                )

        for proc in jobs:
            proc.join()

        if n_permutations:
            return (
                cell_clqs.copy(),
                cell_clqs_adjusted.copy(),
                cell_clqs_permuted.copy(),
            )
        else:
            return cell_clqs.copy()


def clq_single_cell(
    cell_df: DataFrame,
    cat_a: Sequence[Any],
    cat_b: Sequence[Any],
    cat_key: str,
    graph: Optional[spmatrix] = None,
    radius: float = 2,
    n_neighbors: Optional[int] = None,
    n_permutations: Optional[int] = None,
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    seed: int = 808,
    **kwargs: Any,
) -> Union[float, Tuple[float, float, List]]:
    """Compute co-localization quotient (CLQ) for a single cell.

    Parameters
    ----------
    cell_df : DataFrame
        Cell-specific transcript data.
    cat_a : array-like
        First gene category (list of gene names).
    cat_b : array-like
        Second gene category (list of gene names).
    cat_key : str
        Column name in cell_df for gene/category assignment.
    graph : sparse matrix, optional
        Precomputed spatial neighbor graph. If None, constructed from coordinates.
        Default is None.
    radius : float, optional
        Spatial radius for neighbor graph construction. Default is 2.
    n_neighbors : int, optional
        Number of nearest neighbors if radius is None. Default is None.
    n_permutations : int, optional
        Number of permutations for normalized CLQ computation. Default is None.
    x_key : str, optional
        Column name in cell_df for x coordinates. Default is "x".
    y_key : str, optional
        Column name in cell_df for y coordinates. Default is "y".
    z_key : str, optional
        Column name in cell_df for z coordinates. Default is "z".
    seed : int, optional
        Random seed for permutation testing. Default is 808.
    **kwargs
        Additional keyword arguments passed to get_neighbors().

    Returns
    -------
    float or tuple
        If n_permutations is None or 0:
        - clq : float - Raw CLQ score

        If n_permutations > 0:
        - clq : float - Raw CLQ score
        - clq_adjusted : float - Permutation-normalized CLQ score
        - clq_perms : list - Distribution of CLQ scores from permutations
    """

    n_a = (cell_df[cat_key].isin(cat_a)).sum()
    if cat_a == cat_b:
        n_b = n_a - 1
    else:
        n_b = (cell_df[cat_key].isin(cat_b)).sum()
    if n_a == 0 or n_b == 0:  # avoid 0 division
        if n_permutations:
            return 0, np.nan, [0]
        else:
            return 0

    if graph is None:  # build graph from scratch if not provided
        graph = get_neighbors(
            cell_df,
            radius=radius,
            n_neighbors=n_neighbors,
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
            n_jobs=1,  # disable parallel in multiprocessing context
        )
    else:  # in case provided graph includes not only given cell
        graph = graph[np.ix_(cell_df.index.astype(int), cell_df.index.astype(int))]

    cell_df.reset_index(drop=True, inplace=True)  # to ensure index matches graph size
    c_ab = graph[
        np.ix_(
            cell_df[cell_df[cat_key].isin(cat_a)].index,
            cell_df[cell_df[cat_key].isin(cat_b)].index,
        )
    ].sum()

    n = cell_df.shape[0]
    clq = (c_ab / n_a) / (n_b / (n - 1))

    if n_permutations is None or n_permutations == 0:
        return clq

    # permute to have control for score
    else:
        rng = np.random.default_rng(seed)
        clq_perms = []
        for p in range(n_permutations):
            # random shuffling of gene labels
            perm_idx = rng.permutation(cell_df.shape[0])
            permuted_categories = cell_df[cat_key].copy()
            permuted_categories = np.asarray(permuted_categories)[perm_idx]
            cell_df_permuted = cell_df.copy()
            cell_df_permuted[cat_key] = permuted_categories

            # only recompute c_ab, other values are same as unpermuted
            c_ab_permuted = graph[
                np.ix_(
                    cell_df_permuted[cell_df_permuted[cat_key].isin(cat_a)].index,
                    cell_df_permuted[cell_df_permuted[cat_key].isin(cat_b)].index,
                )
            ].sum()

            clq_perms.append((c_ab_permuted / n_a) / (n_b / (n - 1)))

        if np.std(clq_perms) == 0 and np.mean(clq_perms) == 0:
            clq_adjusted = clq
        elif np.std(clq_perms) == 0 and np.mean(clq_perms) == clq:
            clq_adjusted = 0
        else:
            clq_adjusted = clq / np.mean(clq_perms)

        return clq, clq_adjusted, clq_perms


def clq_significance(
    cell_clqs: Dict,
    cell_clqs_permuted: Dict,
    percentile: float = 5,
) -> Tuple[List, Dict]:
    """Assess statistical significance of CLQ scores using permutation distributions.

    Parameters
    ----------
    cell_clqs : dict
        Observed raw CLQ per cell, mapping cell IDs to float values.
    cell_clqs_permuted : dict
        Permutation distributions per cell, mapping cell IDs to lists of
        CLQs from permutations.
    percentile : float, optional
        Percentile threshold for significance testing. A cell is significant if
        its observed CLQ is beyond the [percentile, 100-percentile] range.
        Default is 5.

    Returns
    -------
    tuple
        - significant_clq_cells : list - Cell IDs with significant CLQ scores
        - observed_vs_percentile : dict - Fold-change between observed CLQ and
          the nearest percentile threshold for each cell (1.0 for non-significant)
    """

    observed_vs_percentile = {}
    significant_clq_cells = []
    for cell, clqs in cell_clqs_permuted.items():
        lower_percentile = np.percentile(clqs, percentile)
        upper_percentile = np.percentile(clqs, 100 - percentile)
        observed_clq = cell_clqs[cell]
        if observed_clq < lower_percentile:
            observed_vs_percentile[cell] = lower_percentile / observed_clq
            significant_clq_cells.append(cell)
        elif observed_clq > upper_percentile:
            observed_vs_percentile[cell] = observed_clq / upper_percentile
            significant_clq_cells.append(cell)
        else:
            observed_vs_percentile[cell] = 1

    return significant_clq_cells, observed_vs_percentile
