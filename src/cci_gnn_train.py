import os
import time
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")

H5AD_FILE = os.path.join(DATA_DIR, "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5ad")
LR_FILE = os.path.join(DATA_DIR, "celltalk_human_lr_pair.txt")
TRAIN_EDGES = os.path.join(DATA_DIR, "train_edges.csv")
VAL_EDGES = os.path.join(DATA_DIR, "val_edges.csv")
TEST_EDGES = os.path.join(DATA_DIR, "test_edges.csv")

os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Data Loading & Features ----------

def load_adata(h5ad_path: str):
    print(f"[INFO] Loading AnnData from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    if sparse.issparse(adata.X):
        X = adata.X.tocsr()
    else:
        X = sparse.csr_matrix(adata.X)
    if "spatial" in adata.obsm_keys():
        coords = np.array(adata.obsm["spatial"], dtype=np.float32)
    else:
        print("[WARN] adata.obsm['spatial'] not found. Using zeros.")
        coords = np.zeros((adata.n_obs, 2), dtype=np.float32)
    return adata, X, coords


def read_edges(path: str, require_label: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    expected = ['source', 'target'] + (['label'] if require_label else [])
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"Edges file {path} missing column: {c}")
    return df


def load_lr_gene_sets(lr_path: str, var_names: List[str]):
    df = pd.read_csv(lr_path, sep='\t')
    lig = df['ligand_gene_symbol'].astype(str).tolist()
    rec = df['receptor_gene_symbol'].astype(str).tolist()
    lig_set = sorted(set(lig))
    rec_set = sorted(set(rec))
    var_index = {g: i for i, g in enumerate(var_names)}
    lig_idx = [var_index[g] for g in lig_set if g in var_index]
    rec_idx = [var_index[g] for g in rec_set if g in var_index]
    print(f"[INFO] LR genes mapped: lig={len(lig_idx)} rec={len(rec_idx)}")
    return np.array(lig_idx, dtype=int), np.array(rec_idx, dtype=int)


def build_node_features(X_csr: sparse.csr_matrix, coords: np.ndarray, lig_idx: np.ndarray, rec_idx: np.ndarray):
    # Use LR gene expressions as features, with log1p transform; add spatial coords
    if lig_idx.size > 0:
        lig_expr = X_csr[:, lig_idx].toarray().astype(np.float32)
    else:
        lig_expr = np.zeros((X_csr.shape[0], 0), dtype=np.float32)
    if rec_idx.size > 0:
        rec_expr = X_csr[:, rec_idx].toarray().astype(np.float32)
    else:
        rec_expr = np.zeros((X_csr.shape[0], 0), dtype=np.float32)

    feats = np.concatenate([np.log1p(lig_expr), np.log1p(rec_expr), coords], axis=1)

    # Standardize per feature (z-score)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    feats = (feats - mean) / std
    return feats.astype(np.float32)


# ---------- Graph Construction ----------

def build_knn_adjacency(coords: np.ndarray, k: int = 8):
    n = coords.shape[0]
    print(f"[INFO] Building kNN graph with k={k}")
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k+1)  # include self
    rows = []
    cols = []
    data = []
    for i in range(n):
        neigh = idxs[i, 1:]  # exclude self
        deg = len(neigh)
        if deg == 0:
            continue
        for j in neigh:
            rows.append(i)
            cols.append(j)
            data.append(1.0 / deg)  # row-normalized mean aggregator
    # Make undirected by adding reverse edges (average; still row-normalized per source)
    rows2 = rows + cols
    cols2 = cols + rows
    data2 = data + data
    A = sparse.coo_matrix((data2, (rows2, cols2)), shape=(n, n)).tocsr()
    return A


def scipy_csr_to_torch_sparse(A: sparse.csr_matrix):
    A = A.tocoo()
    indices = torch.tensor(np.vstack((A.row, A.col)), dtype=torch.long)
    values = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(A.shape)).coalesce()


# ---------- Models ----------

class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lin_in1 = nn.Linear(in_dim, hidden_dim)
        self.lin_in2 = nn.Linear(hidden_dim, hidden_dim)
        self.W_self = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.W_neigh = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h = F.relu(self.lin_in1(x))
        h = self.dropout(F.relu(self.lin_in2(h)))
        for l in range(self.num_layers):
            neigh = torch.sparse.mm(A, h)
            h = F.relu(self.W_self[l](h) + self.W_neigh[l](neigh))
            h = self.dropout(h)
        return h


class PairClassifier(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, h: torch.Tensor, pairs: torch.Tensor):
        s = pairs[:, 0].long()
        t = pairs[:, 1].long()
        hs = h[s]
        ht = h[t]
        feat = torch.cat([hs, ht, hs * ht, torch.abs(hs - ht)], dim=1)
        logit = self.net(feat).squeeze(1)
        return logit


# ---------- Training & Evaluation ----------

def train_gnn(node_feats: np.ndarray, A_csr: sparse.csr_matrix,
              train_pairs: np.ndarray, train_labels: np.ndarray,
              val_pairs: np.ndarray, val_labels: np.ndarray,
              hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2,
              lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 50, patience: int = 10):

    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    A = scipy_csr_to_torch_sparse(A_csr).to(device)

    gnn = GraphSAGE(in_dim=node_feats.shape[1], hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    clf = PairClassifier(embed_dim=hidden_dim, dropout=dropout).to(device)

    params = list(gnn.parameters()) + list(clf.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_pairs_t = torch.tensor(train_pairs, dtype=torch.long, device=device)
    train_labels_t = torch.tensor(train_labels, dtype=torch.float32, device=device)
    val_pairs_t = torch.tensor(val_pairs, dtype=torch.long, device=device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.float32, device=device)

    best_aupr = -1.0
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        gnn.train(); clf.train()
        opt.zero_grad()
        h = gnn(x, A)
        logits = clf(h, train_pairs_t)
        loss = loss_fn(logits, train_labels_t)
        loss.backward()
        opt.step()

        # Eval
        gnn.eval(); clf.eval()
        with torch.no_grad():
            h_val = gnn(x, A)
            val_logits = clf(h_val, val_pairs_t)
            val_scores = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_preds = (val_scores >= 0.5).astype(int)

            auroc = roc_auc_score(val_labels, val_scores)
            aupr = average_precision_score(val_labels, val_scores)
            f1 = f1_score(val_labels, val_preds)
            acc = accuracy_score(val_labels, val_preds)

        print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} AUROC={auroc:.4f} AUPR={aupr:.4f} F1={f1:.4f} ACC={acc:.4f}")

        # Early stopping by AUPR
        if aupr > best_aupr:
            best_aupr = aupr
            best_state = {"gnn": gnn.state_dict(), "clf": clf.state_dict()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[INFO] Early stopping at epoch {epoch}")
                break

    # Load best
    if best_state is not None:
        gnn.load_state_dict(best_state["gnn"])
        clf.load_state_dict(best_state["clf"])

    return gnn, clf


def predict_pairs(gnn: GraphSAGE, clf: PairClassifier, node_feats: np.ndarray, A_csr: sparse.csr_matrix, pairs: np.ndarray):
    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    A = scipy_csr_to_torch_sparse(A_csr).to(device)
    gnn.eval(); clf.eval()
    with torch.no_grad():
        h = gnn(x, A)
        pairs_t = torch.tensor(pairs, dtype=torch.long, device=device)
        logits = clf(h, pairs_t)
        scores = torch.sigmoid(logits).cpu().numpy()
    return scores


# ---------- Main ----------

def main():
    t0 = time.time()
    adata, X_csr, coords = load_adata(H5AD_FILE)
    lig_idx, rec_idx = load_lr_gene_sets(LR_FILE, list(adata.var_names))
    node_feats = build_node_features(X_csr, coords.astype(np.float32), lig_idx, rec_idx)

    # Build kNN graph
    A_csr = build_knn_adjacency(coords.astype(np.float32), k=8)

    # Load edges
    train_df = read_edges(TRAIN_EDGES, require_label=True)
    val_df = read_edges(VAL_EDGES, require_label=True)
    test_df = read_edges(TEST_EDGES, require_label=False)

    train_pairs = train_df[['source', 'target']].values.astype(np.int64)
    train_labels = train_df['label'].values.astype(np.int64)
    val_pairs = val_df[['source', 'target']].values.astype(np.int64)
    val_labels = val_df['label'].values.astype(np.int64)
    test_pairs = test_df[['source', 'target']].values.astype(np.int64)

    # Train
    gnn, clf = train_gnn(
        node_feats, A_csr,
        train_pairs, train_labels,
        val_pairs, val_labels,
        hidden_dim=128, num_layers=2, dropout=0.2,
        lr=1e-3, weight_decay=1e-4, epochs=60, patience=10,
    )

    # Validation outputs
    val_scores = predict_pairs(gnn, clf, node_feats, A_csr, val_pairs)
    val_pred = (val_scores >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "aupr": float(average_precision_score(val_labels, val_scores)),
        "f1": float(f1_score(val_labels, val_pred)),
        "accuracy": float(accuracy_score(val_labels, val_pred)),
    }
    print(f"[INFO] Validation metrics: {json.dumps(metrics, indent=2)}")

    val_out = val_df.copy()
    val_out['score'] = val_scores
    val_out['pred'] = val_pred
    val_out_path = os.path.join(OUT_DIR, "val_predictions.csv")
    val_out.to_csv(val_out_path, index=False)

    with open(os.path.join(OUT_DIR, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Test outputs
    test_scores = predict_pairs(gnn, clf, node_feats, A_csr, test_pairs)
    test_out = test_df.copy()
    test_out['score'] = test_scores
    test_out['pred'] = (test_scores >= 0.5).astype(int)
    test_out_path = os.path.join(OUT_DIR, "test_predictions.csv")
    test_out.to_csv(test_out_path, index=False)

    # Submission format
    submission = test_df[['source', 'target']].copy()
    submission['label'] = (test_scores >= 0.5).astype(int)
    submission_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"[INFO] Saved submission CSV: {submission_path}")

    print(f"[INFO] Saved validation predictions: {val_out_path}")
    print(f"[INFO] Saved test predictions: {test_out_path}")
    print(f"[INFO] Done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()