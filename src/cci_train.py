import os
import sys
import time
import json
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
H5AD_FILE = os.path.join(DATA_DIR, "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5ad")
LR_FILE = os.path.join(DATA_DIR, "celltalk_human_lr_pair.txt")
TRAIN_EDGES = os.path.join(DATA_DIR, "train_edges.csv")
VAL_EDGES = os.path.join(DATA_DIR, "val_edges.csv")
TEST_EDGES = os.path.join(DATA_DIR, "test_edges.csv")
OUT_DIR = os.path.join(DATA_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)


def load_adata(h5ad_path: str) -> ad.AnnData:
    print(f"[INFO] Loading AnnData from {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    # Ensure X is CSR for efficient row slicing
    if sparse.issparse(adata.X):
        adata.X = adata.X.tocsr()
    else:
        # convert dense to CSR to standardize operations
        adata.X = sparse.csr_matrix(adata.X)
    if "spatial" in adata.obsm_keys():
        coords = np.array(adata.obsm["spatial"])  # shape (n_obs, 2)
    else:
        print("[WARN] adata.obsm['spatial'] not found. Distances will be set to 0.")
        coords = np.zeros((adata.n_obs, 2), dtype=float)
    return adata, coords


def load_lr_pairs(lr_path: str, adata_var_names: List[str]):
    print(f"[INFO] Loading LR pairs from {lr_path}")
    df = pd.read_csv(lr_path, sep='\t')
    if not {'ligand_gene_symbol', 'receptor_gene_symbol'}.issubset(df.columns):
        raise ValueError("LR file must contain 'ligand_gene_symbol' and 'receptor_gene_symbol' columns")

    var_index = {g: i for i, g in enumerate(adata_var_names)}
    lig_idx, rec_idx, pair_names = [], [], []
    missing_lig, missing_rec = 0, 0
    for _, row in df.iterrows():
        lig = str(row['ligand_gene_symbol'])
        rec = str(row['receptor_gene_symbol'])
        if lig in var_index and rec in var_index:
            lig_idx.append(var_index[lig])
            rec_idx.append(var_index[rec])
            pair_names.append(f"{lig}_{rec}")
        else:
            if lig not in var_index:
                missing_lig += 1
            if rec not in var_index:
                missing_rec += 1
    print(f"[INFO] Filtered LR pairs: {len(pair_names)} usable; missing ligands: {missing_lig}, missing receptors: {missing_rec}")
    lig_idx = np.array(lig_idx, dtype=int)
    rec_idx = np.array(rec_idx, dtype=int)
    return lig_idx, rec_idx, pair_names


def read_edges(path: str, require_label: bool = True):
    df = pd.read_csv(path)
    expected_cols = ['source', 'target'] + (['label'] if require_label else [])
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Edges file {path} missing column: {c}")
    return df


def precompute_row_stats(X: sparse.csr_matrix) -> Tuple[np.ndarray, np.ndarray]:
    # sum per row and nnz per row
    print("[INFO] Precomputing row sums and nonzero counts...")
    row_sum = np.array(X.sum(axis=1)).ravel()
    row_nnz = X.getnnz(axis=1)
    return row_sum, row_nnz


def compute_edge_features(
    X: sparse.csr_matrix,
    coords: np.ndarray,
    s_idx: int,
    t_idx: int,
    lig_idx: np.ndarray,
    rec_idx: np.ndarray,
    row_sum: np.ndarray,
    row_nnz: np.ndarray,
) -> List[float]:
    # ligand expression in source; receptor expression in target
    # use advanced indexing for sparse rows
    s_row = X.getrow(s_idx)
    t_row = X.getrow(t_idx)

    s_lig = s_row[:, lig_idx].toarray().ravel() if lig_idx.size > 0 else np.array([])
    t_rec = t_row[:, rec_idx].toarray().ravel() if rec_idx.size > 0 else np.array([])

    if s_lig.size and t_rec.size:
        prod = s_lig * t_rec
        lr_sum = float(prod.sum())
        lr_mean = float(prod.mean())
        lr_max = float(prod.max())
        lr_nonzero = int(np.count_nonzero(prod > 0))
        # aggregate per-side
        s_lig_sum = float(s_lig.sum())
        t_rec_sum = float(t_rec.sum())
    else:
        lr_sum = lr_mean = lr_max = 0.0
        lr_nonzero = 0
        s_lig_sum = t_rec_sum = 0.0

    # global expression stats
    s_total = float(row_sum[s_idx])
    t_total = float(row_sum[t_idx])
    s_nnz = int(row_nnz[s_idx])
    t_nnz = int(row_nnz[t_idx])

    # spatial distance
    d = float(np.linalg.norm(coords[s_idx] - coords[t_idx])) if coords.size else 0.0

    return [
        lr_sum, lr_mean, lr_max, lr_nonzero,
        s_lig_sum, t_rec_sum,
        s_total, t_total,
        s_nnz, t_nnz,
        d,
    ]


def build_feature_matrix(
    edges_df: pd.DataFrame,
    adata: ad.AnnData,
    coords: np.ndarray,
    lig_idx: np.ndarray,
    rec_idx: np.ndarray,
) -> np.ndarray:
    X = adata.X  # csr
    n_obs = adata.n_obs

    max_idx = int(max(edges_df['source'].max(), edges_df['target'].max()))
    if max_idx >= n_obs:
        raise IndexError(f"Edge index {max_idx} out of bounds for n_obs={n_obs}")

    row_sum, row_nnz = precompute_row_stats(X)

    features = []
    for i, row in edges_df.iterrows():
        s_idx = int(row['source'])
        t_idx = int(row['target'])
        feats = compute_edge_features(X, coords, s_idx, t_idx, lig_idx, rec_idx, row_sum, row_nnz)
        features.append(feats)
    features = np.array(features, dtype=float)
    return features


def train_and_eval(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    print("[INFO] Training RandomForest...")
    clf.fit(X_train_s, y_train)

    val_proba = clf.predict_proba(X_val_s)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, val_proba)),
        "aupr": float(average_precision_score(y_val, val_proba)),
        "f1": float(f1_score(y_val, val_pred)),
        "accuracy": float(accuracy_score(y_val, val_pred)),
    }
    print(f"[INFO] Validation metrics: {json.dumps(metrics, indent=2)}")

    return clf, scaler, metrics


def predict(clf, scaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    proba = clf.predict_proba(Xs)[:, 1]
    return proba


def main():
    t0 = time.time()
    adata, coords = load_adata(H5AD_FILE)
    lig_idx, rec_idx, pair_names = load_lr_pairs(LR_FILE, list(adata.var_names))

    train_df = read_edges(TRAIN_EDGES, require_label=True)
    val_df = read_edges(VAL_EDGES, require_label=True)
    test_df = read_edges(TEST_EDGES, require_label=False)

    print("[INFO] Building features for train/val/test...")
    X_train = build_feature_matrix(train_df, adata, coords, lig_idx, rec_idx)
    y_train = train_df['label'].astype(int).values

    X_val = build_feature_matrix(val_df, adata, coords, lig_idx, rec_idx)
    y_val = val_df['label'].astype(int).values

    X_test = build_feature_matrix(test_df, adata, coords, lig_idx, rec_idx)

    clf, scaler, metrics = train_and_eval(X_train, y_train, X_val, y_val)

    # Save validation predictions and metrics
    val_scores = predict(clf, scaler, X_val)
    val_out = val_df.copy()
    val_out['score'] = val_scores
    val_out['pred'] = (val_scores >= 0.5).astype(int)
    val_out_path = os.path.join(OUT_DIR, "val_predictions.csv")
    val_out.to_csv(val_out_path, index=False)

    with open(os.path.join(OUT_DIR, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Test predictions
    test_scores = predict(clf, scaler, X_test)
    test_out = test_df.copy()
    test_out['score'] = test_scores
    test_out['pred'] = (test_scores >= 0.5).astype(int)
    test_out_path = os.path.join(OUT_DIR, "test_predictions.csv")
    test_out.to_csv(test_out_path, index=False)

    # Submission CSV: source,target,label with 0/1 predictions
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