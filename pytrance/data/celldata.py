from typing import Dict, List, Optional

import numpy as np
from anndata import AnnData
from scipy import sparse as sp
from torch import from_numpy, long, tensor
from torch.utils import data
from torch_geometric.data import Data
from tqdm import tqdm

from ..utils import get_neighbors


class CellData(data.Dataset):
    """PyTorch Dataset for cell-level graph data of transcripts.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with transcript-level observations and features.
    cell_indices : dict
        Dictionary mapping cell IDs to arrays of transcript indices for each cell.
    graph_kwargs : dict, optional
        Keyword arguments for neighbor graph construction passed to get_neighbors().
        Default is None.
    adj : sparse matrix, optional
        Precomputed adjacency matrix. If None, compute per-cell from coordinates.
        Default is None.
    corruption : {'feature_shuffling'}, optional
        Corruption strategy for negative samples in contrastive learning.
        Default is "feature_shuffling".
    seed : int, optional
        Random seed for reproducibility. Default is 808.

    Attributes
    ----------
    data : list
        List of torch_geometric.data.Data objects, one per cell.
    """

    def __init__(
        self,
        adata: AnnData,
        cell_indices: Dict,
        graph_kwargs: Optional[Dict] = None,
        adj: Optional[sp.spmatrix] = None,
        corruption: str = "feature_shuffling",
        seed: int = 808,
    ) -> None:
        self.data = self.__partition__(
            adata, cell_indices, graph_kwargs, adj, corruption, seed
        )

    def __partition__(
        self,
        adata: AnnData,
        cell_indices: Dict,
        graph_kwargs: Optional[Dict],
        adj: Optional[sp.spmatrix],
        corruption: str,
        seed: int,
    ) -> List[Data]:
        """Partition data into per-cell graph objects with features and corruption.

        Parameters
        ----------
        adata : AnnData
            Annotated data object with transcript-level observations and features.
        cell_indices : dict
            Dictionary mapping cell IDs to arrays of transcript indices for each cell.
        graph_kwargs : dict
            Keyword arguments for neighbor graph construction passed to get_neighbors().
        adj : sparse matrix or None
            Precomputed global adjacency matrix. If None, compute per-cell from coordinates.
        corruption : str
            Feature corruption strategy.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list
            List of torch_geometric.data.Data objects with graph, features, and
            corrupted features for each cell.
        """

        data_cells = []
        rng = np.random.default_rng(seed)
        print("setting up torch data set ...")
        for node_idxs in tqdm(cell_indices.values()):
            # construct feature tensor
            cell_adata = adata[node_idxs]
            cell_features_transcripts = cell_adata.X

            # Convert IDs to sparse format
            cell_ids = cell_adata.obs.cell_encoded.values[np.newaxis].T
            spot_ids = node_idxs[np.newaxis].T
            cell_ids_sparse = sp.csr_matrix(cell_ids)
            spot_ids_sparse = sp.csr_matrix(spot_ids)

            # Concatenate
            cell_features = sp.hstack(
                [cell_features_transcripts, cell_ids_sparse, spot_ids_sparse]
            )
            cell_features = from_numpy(cell_features.toarray()).to_sparse()

            # permute features
            if corruption == "feature_shuffling":
                perm_idxs_cell = rng.permutation(np.arange(cell_adata.n_obs))
                corr_fts = cell_features_transcripts[perm_idxs_cell, :]
            else:
                raise ValueError(f"Corruption method '{corruption}' not implemented")

            corr_fts = from_numpy(corr_fts.toarray()).to_sparse()

            if adj is not None:  # use provided graph
                adj_cell = adj[np.ix_(node_idxs, node_idxs)]

            else:  # compute graph from scratch
                adj_cell = get_neighbors(cell_adata.obs, **graph_kwargs)

            # convert adjacency matrix to sparse tensor
            if sp.issparse(adj_cell):
                adj_cell = adj_cell.tocoo()
            else:
                adj_cell = sp.coo_matrix(adj_cell)
            adj_cell = tensor(
                np.vstack((adj_cell.row, adj_cell.col)), dtype=long, device="cpu"
            )

            data_cells.append(
                Data(x=cell_features, edge_index=adj_cell, corr_fts=corr_fts)
            )
        return data_cells

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        out = self.data[index]
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
