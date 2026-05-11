from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from pandas import DataFrame
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA

from .utils import compute_figure_size


def _prepare_hue_coloring(
    cell_df, hue, gene_key, genes, genes_colors, adata=None, cell_key=None, cell=None
):
    """Prepare hue column and color mapping based on hue type."""
    if hue == "gene set":
        cell_df.loc[:, "hue"] = None
        cell_df.loc[cell_df[cell_df[gene_key].isin(genes)].index, "hue"] = "gene set"
        cell_df.loc[cell_df[~cell_df[gene_key].isin(genes)].index, "hue"] = "rest"
        genes_colors["gene set"] = "darkorange"
        genes_colors["rest"] = "lightblue"
        return cell_df, "hue", genes_colors

    elif hue == "gene set only":
        cell_df = cell_df[cell_df[gene_key].isin(genes)]
        return cell_df, cell_df[gene_key].tolist(), genes_colors

    elif hue is None:
        return cell_df, None, genes_colors

    else:
        # hue is a column name - return it as string, not list
        return cell_df, hue, None


def _setup_subplot_axis(
    fig,
    fig_width,
    fig_height,
    row,
    col,
    col_lefts,
    row_bottoms,
    col_widths,
    row_heights,
    subplot_widths,
    subplot_heights,
    df_to_plot,
    boundary_padding,
    x_key,
    y_key,
    cell_boundaries=None,
    nucleus_boundaries=None,
    cell=None,
    technology=None,
    z=0,
):
    """Setup subplot axis with correct positioning and limits."""
    sw = subplot_widths[row][col]
    sh = subplot_heights[row][col]
    grid_cell_w = col_widths[col]
    grid_cell_h = row_heights[row]

    left = col_lefts[col] + (grid_cell_w - sw) / 2
    bottom = row_bottoms[row] + (grid_cell_h - sh) / 2

    norm_left = left / fig_width
    norm_bottom = bottom / fig_height
    norm_width = sw / fig_width
    norm_height = sh / fig_height
    ax = fig.add_axes([norm_left, norm_bottom, norm_width, norm_height])

    # Determine coordinates for axis limits - use boundary if available, otherwise use spots
    if cell_boundaries is not None:
        # Extract boundary coordinates
        if type(cell_boundaries) is list:
            boundary_coords = cell_boundaries[cell]
            x_coords = boundary_coords[:, 0]
            y_coords = boundary_coords[:, 1]
        elif technology == "xenium":
            x_coords = cell_boundaries[cell_boundaries.cell_id == cell].vertex_x.values
            y_coords = cell_boundaries[cell_boundaries.cell_id == cell].vertex_y.values
        elif technology == "merfish":
            mask_x = [
                float(i)
                for i in cell_boundaries.loc[cell, f"boundaryX_z{z}"].split(", ")
            ]
            mask_y = [
                float(i)
                for i in cell_boundaries.loc[cell, f"boundaryY_z{z}"].split(", ")
            ]
            x_coords = np.array(mask_x)
            y_coords = np.array(mask_y)
        else:
            x_coords = np.array(cell_boundaries.loc[cell, x_key])
            y_coords = np.array(cell_boundaries.loc[cell, y_key])

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
    else:
        # Use spot coordinates
        x_min, x_max = df_to_plot[x_key].min(), df_to_plot[x_key].max()
        y_min, y_max = df_to_plot[y_key].min(), df_to_plot[y_key].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if boundary_padding < 1:
        x_pad = x_range * boundary_padding
        y_pad = y_range * boundary_padding
    else:
        x_pad = boundary_padding
        y_pad = boundary_padding

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    return ax


def _set_subplot_title(ax, cell, cell_scores, fontsize):
    """Set subplot title based on cell scores."""
    if cell_scores:
        ax.set_title(f"CLQ_adj = {round(cell_scores[cell], 2)}", fontsize=fontsize)
    else:
        ax.set_title(f"cell: {cell}")


def _plot_transcripts(
    df_to_plot,
    plot_type,
    ax,
    fig,
    gene_key,
    genes,
    hue_cell,
    genes_colors_to_use,
    x_key,
    y_key,
    s,
    cbar=True,
    **kwargs,
):
    """Plot transcripts based on plot type."""
    if plot_type == "scatter":
        sns.scatterplot(
            data=df_to_plot,
            x=x_key,
            y=y_key,
            hue=hue_cell,
            palette=genes_colors_to_use,
            linewidth=0,
            s=s,
            ax=ax,
            legend="full",
        )

        # plot gene set transcripts over rest
        sns.scatterplot(
            data=df_to_plot[df_to_plot[gene_key].isin(genes)],
            x=x_key,
            y=y_key,
            hue=hue_cell,
            palette=genes_colors_to_use,
            linewidth=0,
            s=s,
            ax=ax,
            legend="full",
        )

        ax.set_aspect("equal")

    elif plot_type == "histogram_absolute":
        transcripts_histogram(
            df_to_plot,
            genes,
            ax,
            fig,
            "absolute",
            x_key,
            y_key,
            gene_key,
            cbar=cbar,
            **kwargs,
        )

    elif plot_type == "histogram_relative":
        transcripts_histogram(
            df_to_plot,
            genes,
            ax,
            fig,
            "relative",
            x_key,
            y_key,
            gene_key,
            cbar=cbar,
            **kwargs,
        )


def _collect_subplot_legend(ax, labels_handles, hue, plot_type):
    """Collect and remove legend from subplot."""
    if hue and plot_type == "scatter":
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            labels_handles[label] = handle
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    return labels_handles


def _prepare_plot_data(adata, cells_to_plot, genes, cell_key, gene_key, cmap):
    """Prepare color mappings and grouping for plotting."""
    if genes is None:
        genes = adata.var_names

    # Get colormap and assign colors to genes
    cmap_colors = plt.colormaps[cmap].colors
    genes_in_cells = (
        adata[
            (adata.obs[cell_key].isin(cells_to_plot))
            & (adata.obs[gene_key].isin(genes))
        ]
        .obs[gene_key]
        .unique()
    )
    genes_colors = {
        g: cmap_colors[i % len(cmap_colors)]
        for i, g in enumerate(sorted(genes_in_cells))
    }

    # Group adata by cell
    adata_obs_cell_grouped = adata.obs.groupby(cell_key, observed=False)

    return genes, genes_colors, adata_obs_cell_grouped


def embedding_pca(
    adata: AnnData,
    embeds_pca: np.ndarray,
    key: str = "leiden",
    pca_model: Optional[PCA] = None,
    categorical: bool = True,
    **kwargs: Any,
) -> None:
    """Plot PCA embeddings with gene cluster coloring.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing containing observation metadata.
    embeds_pca : ndarray
        2D PCA-reduced embeddings.
    key : str, optional
        Column name in adata.var for cluster assignments. Default is "leiden".
    pca_model : PCA, optional
        Fitted PCA model. If provided, variance explained percentages are added
        to axis labels. Default is None.
    categorical : bool, optional
        If True, color by categorical clusters using seaborn scatterplot.
        If False, use continuous color mapping. Default is True.
    **kwargs
        Additional keyword arguments passed to scatterplot/scatter functions.

    Returns
    -------
    None
        Displays plot using matplotlib.pyplot.show().
    """

    if categorical:
        scatter = sns.scatterplot(
            x=embeds_pca[:, 0], y=embeds_pca[:, 1], hue=adata.var[key], **kwargs
        )

        scatter.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    else:
        scatter = plt.scatter(
            x=embeds_pca[:, 0], y=embeds_pca[:, 1], c=adata.var[key], **kwargs
        )
        plt.colorbar()

    if pca_model:
        plt.xlabel(f"PC1 ({round(pca_model.explained_variance_[0] * 100, 1)}%)")
        plt.ylabel(f"PC2 ({round(pca_model.explained_variance_[1] * 100, 1)}%)")
    plt.show()

    return


def cells_in_grid(
    n_rows: int,
    n_cols: int,
    adata: AnnData,
    genes: Optional[Sequence[Any]] = None,
    cell_list: Optional[Sequence[Any]] = None,
    cell_scores: Optional[Dict] = None,
    cell_boundaries: Optional[Union[List, np.ndarray, DataFrame]] = None,
    nucleus_boundaries: Optional[Union[List, np.ndarray, DataFrame]] = None,
    hue: str = "gene",
    plot_type: str = "scatter",
    z: Optional[float] = None,
    cell_key: str = "cell",
    gene_key: str = "gene",
    x_key: str = "x",
    y_key: str = "y",
    z_key: str = "z",
    cmap: str = "tab20",
    technology: Optional[str] = None,
    s: float = 50,
    scale: float = 0.1,
    col_padding: float = 10,
    row_padding: float = 10,
    boundary_padding: float = 5,
    fontsize: int = 14,
    return_fig: bool = False,
    cbar: bool = True,
    **kwargs: Any,
) -> Optional[Figure]:
    """Create multi-panel grid visualization of cells with spots/transcripts.

    Parameters
    ----------
    n_rows : int
        Number of rows in subplot grid.
    n_cols : int
        Number of columns in subplot grid.
    adata : AnnData
        Annotated data object with transcript-level observations.
    genes : array-like, optional
        Genes to highlight. If None, all genes in adata are used. Default is None.
    cell_list : array-like
        List of cell IDs to plot.
    cell_scores : dict, optional
        Co-localization scores per cell for subplot titles. Default is None.
    cell_scores : dict, optional
        (Adjusted) scores per cell for titles. Default is None.
    cell_boundaries : list, ndarray, or DataFrame, optional
        Cell boundary coordinates (format depends on technology). Default is None.
    nucleus_boundaries : list, ndarray, or DataFrame, optional
        Nucleus boundary coordinates (format depends on technology). Default is None.
    hue : str, optional
        Style to use for coloring transcripts ('gene', 'gene set', 'gene set only',
        or column name). Default is "gene".
    plot_type : {'scatter', 'histogram_absolute', 'histogram_relative'}, optional
        Type of plot for each subplot. Default is "scatter".
    z : float, optional
        Z-slice value to filter transcripts. If None, uses all z values. Default is None.
    cell_key : str, optional
        Column name in adata.obs for cell IDs. Default is "cell".
    gene_key : str, optional
        Column name in adata.obs for gene assignment. Default is "gene".
    x_key : str, optional
        Column name in adata.obs for x coordinates. Default is "x".
    y_key : str, optional
        Column name in adata.obs for y coordinates. Default is "y".
    z_key : str, optional
        Column name in adata.obs for z coordinates. Default is "z".
    cmap : str, optional
        Colormap name for gene coloring (scatter plots). Default is "tab20".
    technology : {'xenium', 'merfish'}, optional
        Technology for boundary format interpretation. Default is None.
    s : float, optional
        Marker size for scatter plots. Default is 50.
    scale : float, optional
        Scale factor for subplot dimensions. Default is 0.1.
    col_padding : float, optional
        Horizontal padding between columns. Default is 10.
    row_padding : float, optional
        Vertical padding between rows. Default is 10.
    boundary_padding : float, optional
        Padding around data in axis limits. Default is 5.
    fontsize : int, optional
        Font size for subplot titles. Default is 14.
    return_fig : bool, optional
        If True, return figure object. If False, display with plt.show(). Default is False.
    cbar : bool, optional
        Show colorbar for histogram plots. Default is True.
    **kwargs
        Additional keyword arguments passed to plotting functions.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object if return_fig=True, otherwise None.
    """

    n_cells = n_rows * n_cols
    if cell_list is None or len(cell_list) == 0:
        raise ValueError("no cells specified")
    cells_to_plot = cell_list[:n_cells]

    # Prepare plot data (genes, colors, grouping)
    # For scatter plots, use provided cmap for gene coloring; for histograms, use neutral default
    genes_colors_cmap = cmap if not plot_type.startswith("histogram") else "tab20"
    genes, genes_colors, adata_obs_cell_grouped = _prepare_plot_data(
        adata, cells_to_plot, genes, cell_key, gene_key, genes_colors_cmap
    )
    labels_handles = {}

    # Pass histogram cmap through kwargs only if custom cmap provided (not default 'tab20')
    if plot_type.startswith("histogram") and cmap != "tab20" and "cmap" not in kwargs:
        kwargs["cmap"] = cmap

    # compute figure size after scaling
    (
        fig_width,
        fig_height,
        row_heights,
        col_widths,
        subplot_heights,
        subplot_widths,
        col_padding,
        row_padding,
    ) = compute_figure_size(
        adata,
        cell_list,
        n_rows,
        n_cols,
        scale=scale,
        col_padding=col_padding,
        row_padding=row_padding,
        boundary_padding=boundary_padding,
        cell_key=cell_key,
        x_key=x_key,
        y_key=y_key,
    )
    fig = plt.figure(figsize=(fig_width, fig_height))

    cell_count = 0

    col_lefts = [sum(col_widths[:col]) + col * col_padding for col in range(n_cols)]
    row_bottoms = [
        sum(row_heights[row + 1 :]) + (n_rows - row - 1) * row_padding
        for row in range(n_rows)
    ]

    for row in range(n_rows):
        for col in range(n_cols):
            cell = cells_to_plot[cell_count]
            cell_df = adata_obs_cell_grouped.get_group(cell).copy()
            df_to_plot = cell_df

            # Prepare hue coloring
            df_to_plot, hue_cell, genes_colors_to_use = _prepare_hue_coloring(
                df_to_plot,
                hue,
                gene_key,
                genes,
                genes_colors.copy(),
                adata=adata,
                cell_key=cell_key,
                cell=cell,
            )

            # only take transcripts from given z slice
            if z:
                df_to_plot = df_to_plot[df_to_plot[z_key] == z]

            # Setup subplot axis with proper positioning and limits
            ax = _setup_subplot_axis(
                fig,
                fig_width,
                fig_height,
                row,
                col,
                col_lefts,
                row_bottoms,
                col_widths,
                row_heights,
                subplot_widths,
                subplot_heights,
                df_to_plot,
                boundary_padding,
                x_key,
                y_key,
                cell_boundaries=cell_boundaries,
                nucleus_boundaries=nucleus_boundaries,
                cell=cell,
                technology=technology,
                z=z,
            )

            # Plot transcripts
            _plot_transcripts(
                df_to_plot,
                plot_type,
                ax,
                fig,
                gene_key,
                genes,
                hue_cell,
                genes_colors_to_use,
                x_key,
                y_key,
                s,
                cbar=cbar,
                **kwargs,
            )

            # Set title
            _set_subplot_title(ax, cell, cell_scores, fontsize)

            if cell_boundaries is not None:
                mask(
                    cell_boundaries,
                    cell,
                    ax,
                    technology=technology,
                    x_key=x_key,
                    y_key=y_key,
                )
            if nucleus_boundaries is not None:
                mask(
                    nucleus_boundaries,
                    cell,
                    ax,
                    technology=technology,
                    x_key=x_key,
                    y_key=y_key,
                )

            cell_count += 1

            # Collect legend from subplot
            labels_handles = _collect_subplot_legend(ax, labels_handles, hue, plot_type)

    if hue and plot_type == "scatter":
        labels_handles = dict(sorted(labels_handles.items()))
        fig.legend(
            labels_handles.values(),
            labels_handles.keys(),
            handler_map=genes_colors,
            loc="center left",
            bbox_to_anchor=(1.1, 0.5),
            fontsize=fontsize,
            markerscale=2,
        )
    # plt.show()

    if return_fig:
        return fig
    else:
        return


def mask(
    boundaries: Union[List, np.ndarray, DataFrame],
    cell: Any,
    ax: Axes,
    technology: Optional[str] = None,
    z: float = 0,
    x_key: str = "x",
    y_key: str = "y",
    c: str = "black",
    **kwargs: Any,
) -> None:
    """Draw cell or nucleus boundary mask on matplotlib axis.

    Parameters
    ----------
    boundaries : list, ndarray, or DataFrame
        Boundary coordinates. Format depends on technology type.
    cell : str
        Cell ID to retrieve boundaries for.
    ax : matplotlib.axes.Axes
        Target axis to draw on.
    technology : {'xenium', 'merfish'}, optional
        Technology for boundary format interpretation. Default is None.
    z : float, optional
        Z-slice for MERFISH boundaries. Default is 0.
    x_key : str, optional
        Column name for x coordinates in DataFrame. Default is "x".
    y_key : str, optional
        Column name for y coordinates in DataFrame. Default is "y".
    c : str, optional
        Line color. Default is "black".
    **kwargs
        Additional keyword arguments passed to ax.plot().

    Returns
    -------
    None
    """

    if type(boundaries) is list:
        ax.plot(*boundaries[cell].T, c=c, **kwargs)
    elif technology == "xenium":
        ax.plot(
            boundaries[boundaries.cell_id == cell].vertex_x,
            boundaries[boundaries.cell_id == cell].vertex_y,
            c=c,
            **kwargs,
        )
    elif technology == "merfish":
        if (
            type(boundaries.loc[cell, f"boundaryX_z{z}"]) is not str
        ):  # skip if no mask at this z
            return
        mask_x = [float(i) for i in boundaries.loc[cell, f"boundaryX_z{z}"].split(", ")]
        mask_y = [float(i) for i in boundaries.loc[cell, f"boundaryY_z{z}"].split(", ")]
        mask = np.vstack((mask_x, mask_y))
        ax.plot(*mask, c=c, **kwargs)
    else:
        mask_x = [i for i in boundaries.loc[cell, x_key]]
        mask_y = [i for i in boundaries.loc[cell, y_key]]
        mask = np.vstack((mask_x, mask_y))
        ax.plot(*mask, c=c, **kwargs)

    return


def transcripts_histogram(
    cell_df: DataFrame,
    genes: Sequence[Any],
    ax: Axes,
    fig: Figure,
    type: str = "absolute",
    x_key: str = "x",
    y_key: str = "y",
    gene_key: str = "gene",
    bin_size: float = 25,
    vmax: Optional[float] = None,
    return_fig: bool = False,
    discrete: bool = False,
    cbar: bool = True,
    **kwargs: Any,
) -> Optional[AxesImage]:
    """Create 2D histogram visualization of transcript density.

    Parameters
    ----------
    cell_df : DataFrame
        Cell transcript data.
    genes : array-like
        Genes to include in histogram.
    ax : matplotlib.axes.Axes
        Target axis for plotting.
    fig : matplotlib.figure.Figure
        Figure object for adding colorbar.
    type : {'absolute', 'relative', 'normalized'}, optional
        Histogram type. 'absolute': counts, 'relative': counts/total,
        'normalized': L2-normalized counts. Default is \"absolute\".
    x_key : str, optional
        Column name for x coordinates. Default is "x".
    y_key : str, optional
        Column name for y coordinates. Default is "y".
    gene_key : str, optional
        Column name for gene assignment. Default is "gene".
    bin_size : float, optional
        Size of histogram bins. Default is 25.
    vmax : float, optional
        Maximum value for color scale. Only used for 'relative' type. Default is None.
    return_fig : bool, optional
        If True, return image object. Default is False.
    discrete : bool, optional
        If True, use discrete colors for integer counts. Default is False.
    cbar : bool, optional
        Show colorbar. Default is True.
    **kwargs
        Additional keyword arguments passed to ax.imshow() (e.g., cmap).

    Returns
    -------
    matplotlib.image.AxesImage or None
        Image object if return_fig=True, otherwise None.
    """

    cell_df_gene_group = cell_df[cell_df[gene_key].isin(genes)]

    if "cmap" not in kwargs.keys():
        n_colors = 256
        color_array = plt.colormaps.get_cmap("Oranges")(range(n_colors))
        color_array[0, -1] = 0
        alpha_oranges_cmap = LinearSegmentedColormap.from_list("", colors=color_array)
        kwargs["cmap"] = alpha_oranges_cmap

    bin_edges_x = np.arange(cell_df[x_key].min(), cell_df[x_key].max(), bin_size)
    bin_edges_y = np.arange(cell_df[y_key].min(), cell_df[y_key].max(), bin_size)

    hist_gg, xedges_gg, yedges_gg = np.histogram2d(
        x=cell_df_gene_group[x_key].values,
        y=cell_df_gene_group[y_key].values,
        bins=[bin_edges_x, bin_edges_y],
    )
    if discrete:
        norm = BoundaryNorm(
            list(range(int(np.max(hist_gg)) + 1)), kwargs["cmap"].N, extend="max"
        )
        kwargs["norm"] = norm
    if type == "absolute":
        hist = ax.imshow(
            hist_gg.T,
            interpolation="nearest",
            origin="lower",
            extent=[xedges_gg[0], xedges_gg[-1], yedges_gg[0], yedges_gg[-1]],
            # alpha=0.7,
            aspect="equal",
            **kwargs,
        )
    elif type == "relative":
        hist_total, xedges_total, yedges_total = np.histogram2d(
            x=cell_df[x_key].values,
            y=cell_df[y_key].values,
            bins=[bin_edges_x, bin_edges_y],
        )

        hist = ax.imshow(
            np.nan_to_num(
                np.divide(
                    hist_gg,
                    hist_total,
                    out=np.zeros_like(hist_gg),
                    where=hist_total != 0,
                ),
                0,
            ).T,
            interpolation="nearest",
            origin="lower",
            extent=[xedges_gg[0], xedges_gg[-1], yedges_gg[0], yedges_gg[-1]],
            vmax=vmax,
            aspect="equal",
            **kwargs,
        )

    elif type == "normalized":
        hist_norm = hist_gg / np.linalg.norm(hist_gg)
        hist = ax.imshow(
            hist_norm.T,
            interpolation="nearest",
            origin="lower",
            extent=[xedges_gg[0], xedges_gg[-1], yedges_gg[0], yedges_gg[-1]],
            # alpha=0.7,
            aspect="equal",
            **kwargs,
        )

    if cbar:
        fig.colorbar(hist, ax=ax)
    if return_fig:
        return hist
    else:
        return


def dendrogram(
    model: Any,
    gene_names_ordered: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Dict:
    """Plot hierarchical clustering dendrogram from model.

    Parameters
    ----------
    model : object
        Fitted hierarchical clustering model.
    gene_names_ordered : array-like, optional
        Ordered gene names for x-axis labels. If None, numeric indices are used.
        Default is None.
    **kwargs
        Additional keyword arguments passed to scipy.cluster.hierarchy.dendrogram().

    Returns
    -------
    dict
        Dendrogram dictionary from scipy.cluster.hierarchy.dendrogram().
    """

    # Create linkage matrix and then plot the dendrogram

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

    # Plot the corresponding dendrogram
    fig, ax = plt.subplots()
    dendro = hierarchy.dendrogram(linkage_matrix, **kwargs)
    if gene_names_ordered:
        ax.set_xticklabels(gene_names_ordered, rotation=90)
    plt.show()

    return dendro


def pairwise_clq_heatmap(
    gene_set: Sequence[Any],
    clq_scores: Dict,
    return_matrix: bool = False,
    return_axis: bool = False,
    **kwargs: Any,
) -> Union[DataFrame, Axes, Tuple[DataFrame, Axes], None]:
    """Visualize pairwise CLQ scores as heatmap.

    Parameters
    ----------
    gene_set : array-like
        List of gene names to include in the heatmap.
    clq_scores : dict
        Dictionary mapping gene pairs (tuples) to CLQ score values.
        Expected keys are (gene1, gene2) tuples for all pairs in gene_set.
    return_matrix : bool, optional
        If True, return the CLQ score matrix. Default is False.
    return_axis : bool, optional
        If True, return the heatmap axis/axes object. Default is False.
    **kwargs
        Additional keyword arguments passed to seaborn.heatmap().

    Returns
    -------
    DataFrame or AxesSubplot or tuple or None
        Depending on return_matrix and return_axis flags:
        - (False, False): None (displays plot)
        - (True, False): DataFrame with CLQ scores
        - (False, True): Heatmap axes object
        - (True, True): Tuple of (DataFrame, heatmap axes)
    """

    clq_adj_matrix = np.zeros((len(gene_set), len(gene_set)))
    for i, gene1 in enumerate(gene_set):
        for j, gene2 in enumerate(gene_set):
            clq_adj_matrix[i, j] = round(clq_scores[(gene1, gene2)], 2)

    clq_adj_df = DataFrame(clq_adj_matrix, columns=gene_set, index=gene_set)

    fig, ax = plt.subplots()
    heatmap = sns.heatmap(clq_adj_df, annot=True, fmt=".1f", vmin=0, ax=ax, **kwargs)

    if return_matrix:
        if return_axis:
            return clq_adj_df, heatmap
        return clq_adj_df
    else:
        if return_axis:
            return heatmap
        else:
            return
