import os
import random
import time
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import get_neighbors, sparse_mx_to_torch_sparse_tensor


def seed_everything(seed: int) -> None:
    """Seed supported random-number generators and deterministic backends."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    layer_type: str = "gcn",
    sparse: bool = True,
    corruption: str = "feature_shuffling",
    seed: int = 808,
) -> Tuple[float, np.ndarray]:
    """Train model for one epoch with contrastive learning.

    Parameters
    ----------
    model : torch.nn.Module
        Graph neural network model to train.
    train_loader : DataLoader
        PyTorch DataLoader providing batches of graph data.
    optimiser : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device (CPU/GPU) to run training on.
    layer_type : str
        Type of layer Default is "gcn".
    sparse : bool
        Whether to use sparse adjacency matrix representation. Default is True
    corruption : str, optional
        Corruption strategy for negative samples ('feature_shuffling' or
        'adjacency_shuffling'). Default is "feature_shuffling".
    seed : int, optional
        Random seed for reproducibility in corruption. Default is 808.

    Returns
    -------
    tuple
        - avg_loss : float - Average training loss over all batches
        - spot_ids : ndarray - Spot/node IDs from the last batch
    """

    model.train()

    total_loss = 0
    total_nodes = 0
    corruption_rng = np.random.RandomState(seed)
    for batch in tqdm(train_loader, desc="Processing batches", leave=False):
        batch.x = batch.x.to_dense()
        spot_ids = batch.x[:, -1]
        batch.x = batch.x[:, :-2].to(device, torch.float32)
        corr_fts = batch.corr_fts.to_dense().to(device, torch.float32)
        optimiser.zero_grad()

        nb_nodes_batch = batch.x.shape[0]
        lbl_1 = torch.ones(1, nb_nodes_batch)
        lbl_2 = torch.zeros(1, nb_nodes_batch)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(device)

        if sparse:
            data = np.ones((len(batch.edge_index[1])))
            row = batch.edge_index[0]
            col = batch.edge_index[1]
            shape = (batch.x.shape[0], batch.x.shape[0])
            sp_adj_batch = sparse_mx_to_torch_sparse_tensor(
                sp.coo_matrix((data, (row, col)), shape=shape)
            ).to(device)
            corr_sp_adj_batch = sp_adj_batch
            if corruption == "adjacency_shuffling":
                idx = corruption_rng.permutation(len(row))
                corr_row = row[idx]
                corr_sp_adj_batch = sparse_mx_to_torch_sparse_tensor(
                    sp.coo_matrix((data, (corr_row, col)), shape=shape)
                ).to(device)
                corr_fts = batch.x

        if layer_type != "gcn":
            sp_adj_batch = sp_adj_batch.to_sparse_csr()
            corr_sp_adj_batch = corr_sp_adj_batch.to_sparse_csr()

        logits = model(
            batch.x,
            sp_adj_batch if sparse else batch.edge_index,
            corr_fts,
            corr_sp_adj_batch,
            sparse,
            None,
            None,
            None,
        )
        loss = nn.BCEWithLogitsLoss()(logits, lbl)  # binary cross entropy

        loss.backward()
        optimiser.step()

        nodes = batch.x.shape[0]
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes, spot_ids  # , perm_idxs_batch


def compute_embeddings(
    model: nn.Module,
    train_loader: DataLoader,
    embeds_shape: Tuple[int, ...],
    layer_type: str = "gcn",
) -> np.ndarray:
    """Compute spot embeddings using trained model.

    Parameters
    ----------
    model : torch.nn.Module
        Trained graph neural network model.
    train_loader : DataLoader
        PyTorch DataLoader providing batches of graph data.
    embeds_shape : tuple
        Expected output shape for embedding array.
    layer_type : str, optional
        Type of graph layer ('gcn' or other). Affects adjacency matrix format.
        Default is "gcn".

    Returns
    -------
    ndarray
        Spot embeddings of shape embeds_shape with one embedding per spot.
        Ordering corresponds to spot_ids extracted from batches.
    """

    model.eval()

    embeds = np.zeros(embeds_shape)
    for batch in train_loader:
        batch.x = batch.x.to_dense()
        spot_ids = batch.x[:, -1].cpu().numpy().astype(int)
        batch.x = batch.x[:, :-2].to("cuda", torch.float32)

        data = np.ones((len(batch.edge_index[1])))
        row = batch.edge_index[0]
        col = batch.edge_index[1]
        shape = (batch.x.shape[0], batch.x.shape[0])
        sp_adj_batch = sparse_mx_to_torch_sparse_tensor(
            sp.coo_matrix((data, (row, col)), shape=shape)
        ).to("cuda")
        if layer_type != "gcn":
            sp_adj_batch = sp_adj_batch.coalesce()
        train_embeds_batch, logits_batch = model.embed(
            batch.x, sp_adj_batch, True, None
        )
        train_embeds_batch = train_embeds_batch.cpu().numpy().squeeze()
        for j, id in enumerate(spot_ids):
            embeds[id] = train_embeds_batch[j]

    return embeds


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    embeds_shape: Tuple[int, ...],
    n_epochs: int,
    sparse: bool = True,
    layer_type: str = "gcn",
    dir_path: Optional[str] = None,
    corruption: str = "feature_shuffling",
    save_steps: int = 10,
    save_final: bool = True,
    seed: int = 808,
) -> None:
    """Train graph neural network model with contrastive learning.

    Parameters
    ----------
    model : torch.nn.Module
        Graph neural network model to train.
    train_loader : DataLoader
        PyTorch DataLoader providing batches of graph data.
    optimiser : torch.optim.Optimizer
        Optimizer for updating model parameters.
    device : torch.device
        Device (CPU/GPU) to run training on.
    embeds_shape : tuple
        Expected output shape for embedding array.
    n_epochs : int
        Number of training epochs.
    sparse : bool
        Whether to use sparse adjacency matrix representation. Default is True.
    layer_type : str
        Type of graph layer. Default is "gcn".
    dir_path : str, optional
        Directory path for saving results. If None, uses current working directory.
        Default is None.
    corruption : str, optional
        Corruption strategy for negative samples. Default is \"feature_shuffling\".
    save_steps : int, optional
        Number of checkpoints to save evenly spaced throughout training.
        Default is 10.
    save_final : bool, optional
        Whether to compute and save the embeddings after training. Default is True.
    seed : int, optional
        Random seed used for reproducible training. Default is 808.

    Returns
    -------
    None
        Saves training artifacts to disk including model checkpoints, embeddings,
        and TensorBoard event files.
    """

    seed_everything(seed)

    if dir_path is None:  # use directory from which code is run
        dir_path = os.getcwd() + "/"
    print(f"Save results to: {dir_path}")

    writer = SummaryWriter(log_dir=dir_path)
    train_losses = []
    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, sids = train_epoch(
            model=model,
            train_loader=train_loader,
            optimiser=optimiser,
            sparse=sparse,
            device=device,
            layer_type=layer_type,
            corruption=corruption,
            seed=seed,
        )

        train_losses.append(train_loss)
        writer.add_scalar("train loss", train_loss, epoch)
        end_time = time.time()
        passed_time = np.round(end_time - start_time)

        print(f"Epoch: {epoch}\ttrain loss: {train_loss}\tpassed time: {passed_time}s")

        if save_steps:  # save intermediate results
            if epoch % (n_epochs / save_steps) == 0 and epoch > 0:
                print("saving embeddings ... ")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimiser_state_dict": optimiser.state_dict(),
                        "train loss": train_loss,
                    },
                    dir_path + f"{epoch}epochs.pt",
                )

                embeds = compute_embeddings(
                    model, train_loader, embeds_shape, layer_type
                )
                np.savez_compressed(dir_path + "embeds.npz", a=embeds)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "train loss": train_loss,
        },
        dir_path + "final.pt",
    )
    writer.flush()
    writer.close()

    # save final embeddings after full training
    if save_final:
        print("saving embeddings ... ")
        embeds = compute_embeddings(model, train_loader, embeds_shape, layer_type)
        np.savez_compressed(dir_path + "embeds.npz", a=embeds)


def neighborhood_aggregation_single_cell(
    node_features: np.ndarray,
    adj_cell: sp.spmatrix,
) -> np.ndarray:
    """Aggregate node features over a sparse adjacency matrix.

    Parameters
    ----------
    node_features : ndarray
        Feature matrix with one row per node.
    adj_cell : scipy.sparse.spmatrix
        Sparse adjacency matrix describing the neighbors of each node.

    Returns
    -------
    ndarray
        Aggregated feature matrix with one row per node.
    """
    if adj_cell.nnz == 0:
        return np.zeros_like(node_features, dtype=np.float32)
    return adj_cell @ node_features


def neighborhood_aggregation(
    transcripts_df: pd.DataFrame,
    graph_kwargs: dict[str, Any],
    cell_key: str = "cell",
    gene_key: str = "gene",
    x_key: str = "x",
    y_key: str = "y",
    z_key: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate transcript features over spatial transcript neighbors.

    Parameters
    ----------
    transcripts_df : DataFrame
        Transcript table containing cell, gene, and coordinate columns.
    graph_kwargs : dict
        Keyword arguments passed to :func:`get_neighbors` when constructing
        each cell's spatial graph.
    cell_key : str, optional
        Column identifying cells. Default is "cell".
    gene_key : str, optional
        Column identifying genes. Default is "gene".
    x_key : str, optional
        Column containing x coordinates. Default is "x".
    y_key : str, optional
        Column containing y coordinates. Default is "y".
    z_key : str, optional
        Column containing z coordinates. If provided, a 3D spatial graph is
        constructed and the z coordinate is included in the output. Default
        is None.

    Returns
    -------
    DataFrame
        Transcript table containing the original cell, gene, and coordinate
        columns followed by one neighbor-aggregation feature column per gene.
    """
    if z_key is not None:
        graph_kwargs = {**graph_kwargs, "z_key": z_key}

    coordinate_keys = [x_key, y_key]
    if z_key is not None:
        coordinate_keys.append(z_key)

    gene_names = sorted(transcripts_df[gene_key].astype(str).dropna().unique())
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    feature_matrix = np.zeros((len(transcripts_df), len(gene_names)), dtype=np.int64)
    
    for cell_id, cell_df in tqdm(transcripts_df.groupby(cell_key, sort=False)):
        cell_idx = cell_df.index.astype(int).to_numpy()
        if len(cell_df) < 2:
            continue

        adj_cell = get_neighbors(cell_df, **graph_kwargs).tocsr()

        cell_gene_names = cell_df[gene_key].astype(str).to_numpy()
        cell_indicator = np.zeros((len(cell_df), len(gene_names)), dtype=np.float32)
        cell_gene_idx = np.fromiter(
            (gene_to_idx.get(gene_name, -1) for gene_name in cell_gene_names),
            dtype=np.int64,
            count=len(cell_gene_names),
        )
        valid = cell_gene_idx >= 0
        if np.any(valid):
            cell_indicator[np.arange(len(cell_df))[valid], cell_gene_idx[valid]] = 1.0

        neighbor_features = neighborhood_aggregation_single_cell(cell_indicator, adj_cell)
        feature_matrix[cell_idx] = neighbor_features.astype(np.int64)

    feature_df = pd.DataFrame(
        feature_matrix,
        columns=gene_names,
        index=transcripts_df.index,
    )
    feature_df = pd.concat(
        [
            transcripts_df[[cell_key, gene_key] + coordinate_keys].reset_index(
                drop=True
            ),
            feature_df.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_df = feature_df[[cell_key, gene_key] + coordinate_keys + gene_names]
    
    return feature_df