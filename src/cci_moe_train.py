import os
import time
import json
import argparse
import numpy as np
import pandas as pd
from typing import List
from scipy import sparse
from scipy.spatial import cKDTree
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from torch_geometric.nn import SAGEConv, GATConv, TransformerConv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUT_DIR = os.path.join(ROOT_DIR, "outputs")
H5AD_FILE = os.path.join(DATA_DIR, "Visium_Human_Breast_Cancer_filtered_feature_bc_matrix.h5ad")
LR_FILE = os.path.join(DATA_DIR, "celltalk_human_lr_pair.txt")
TRAIN_EDGES = os.path.join(DATA_DIR, "train_edges.csv")
VAL_EDGES = os.path.join(DATA_DIR, "val_edges.csv")
TEST_EDGES = os.path.join(DATA_DIR, "test_edges.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Data & Features ----------

def load_adata(path: str):
    adata = ad.read_h5ad(path)
    X = adata.X.tocsr() if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    coords = np.array(adata.obsm.get("spatial", np.zeros((adata.n_obs, 2), dtype=np.float32)), dtype=np.float32)
    return adata, X, coords


def load_lr_gene_sets(lr_path: str, var_names: List[str]):
    df = pd.read_csv(lr_path, sep='\t')
    lig = df['ligand_gene_symbol'].astype(str).tolist()
    rec = df['receptor_gene_symbol'].astype(str).tolist()
    lig_set, rec_set = sorted(set(lig)), sorted(set(rec))
    var_index = {g: i for i, g in enumerate(var_names)}
    lig_idx = [var_index[g] for g in lig_set if g in var_index]
    rec_idx = [var_index[g] for g in rec_set if g in var_index]
    return np.array(lig_idx, dtype=int), np.array(rec_idx, dtype=int)


def build_node_features(X_csr: sparse.csr_matrix, coords: np.ndarray, lig_idx: np.ndarray, rec_idx: np.ndarray):
    lig = X_csr[:, lig_idx].toarray().astype(np.float32) if lig_idx.size > 0 else np.zeros((X_csr.shape[0], 0), dtype=np.float32)
    rec = X_csr[:, rec_idx].toarray().astype(np.float32) if rec_idx.size > 0 else np.zeros((X_csr.shape[0], 0), dtype=np.float32)
    feats = np.concatenate([np.log1p(lig), np.log1p(rec), coords], axis=1)
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    return ((feats - mean) / std).astype(np.float32)

# ---------- Graphs ----------

def build_knn_adjacency(coords: np.ndarray, k: int = 10):
    n = coords.shape[0]
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords, k=k+1)
    rows, cols = [], []
    for i in range(n):
        neigh = idxs[i, 1:]
        for j in neigh:
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    return A


def adj_to_edge_index(A: sparse.csr_matrix):
    A = A.tocoo()
    return np.vstack([A.row.astype(np.int64), A.col.astype(np.int64)])


def build_lr_edges_simple(X_csr: sparse.csr_matrix, lig_idx: np.ndarray, rec_idx: np.ndarray, top_m: int = 8, min_strength: float = 0.0, symmetric: bool = True):
    n = X_csr.shape[0]
    L_sum = np.array(X_csr[:, lig_idx].sum(axis=1)).ravel() if lig_idx.size > 0 else np.zeros(n, dtype=np.float32)
    R_sum = np.array(X_csr[:, rec_idx].sum(axis=1)).ravel() if rec_idx.size > 0 else np.zeros(n, dtype=np.float32)
    edges = set()
    for i in range(n):
        scores = L_sum[i] * R_sum
        scores[i] = -np.inf
        idx = np.argpartition(-scores, top_m)[:top_m] if top_m < n else np.argsort(-scores)
        for j in idx:
            if scores[j] >= min_strength:
                edges.add((i, int(j)))
                if symmetric:
                    edges.add((int(j), i))
    return np.array(list(edges), dtype=np.int64).T if edges else np.empty((2,0), dtype=np.int64)


def build_edge_attr(edge_index: np.ndarray, coords: np.ndarray, L_sum: np.ndarray, R_sum: np.ndarray):
    s, t = edge_index[0], edge_index[1]
    dist = np.linalg.norm(coords[s] - coords[t], axis=1)
    lr_strength = L_sum[s] * R_sum[t]
    F = np.stack([dist, lr_strength], axis=1).astype(np.float32)
    m = F.mean(axis=0, keepdims=True); sd = F.std(axis=0, keepdims=True) + 1e-6
    return (F - m) / sd

# ---------- Experts ----------

class SAGEEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)] + [SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers-1)])
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = hidden_dim
    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)); h = self.dropout(h)
        return h

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, heads=4, dropout=0.2, concat=False):
        super().__init__()
        self.convs = nn.ModuleList([GATConv(in_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout)])
        in_next = hidden_dim * heads if concat else hidden_dim
        for _ in range(num_layers-1):
            self.convs.append(GATConv(in_next, hidden_dim, heads=heads, concat=concat, dropout=dropout))
            in_next = hidden_dim * heads if concat else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = in_next
    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index)); h = self.dropout(h)
        return h

class TransEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, num_layers=2, heads=4, dropout=0.2, concat=False):
        super().__init__()
        self.convs = nn.ModuleList([TransformerConv(in_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout)])
        in_next = hidden_dim * heads if concat else hidden_dim
        for _ in range(num_layers-1):
            self.convs.append(TransformerConv(in_next, hidden_dim, heads=heads, concat=concat, dropout=dropout))
            in_next = hidden_dim * heads if concat else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = in_next
    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index, edge_attr)); h = self.dropout(h)
        return h

class PairClassifier(nn.Module):
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim*4, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    def forward(self, h, pairs):
        s, t = pairs[:,0].long(), pairs[:,1].long()
        hs, ht = h[s], h[t]
        feat = torch.cat([hs, ht, hs*ht, torch.abs(hs-ht)], dim=1)
        return self.net(feat).squeeze(1)

class GatingNet(nn.Module):
    def __init__(self, in_dim=4, num_experts=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, num_experts))
    def forward(self, x):
        return F.softmax(self.net(x), dim=1)

# ---------- Utils ----------

def read_edges(path: str, require_label: bool):
    df = pd.read_csv(path)
    cols = ['source','target'] + (['label'] if require_label else [])
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    return df

def build_degrees(n_nodes: int, edge_index: np.ndarray):
    deg = np.bincount(edge_index[0], minlength=n_nodes)
    return deg.astype(np.int64)

def gating_features(pairs: np.ndarray, coords: np.ndarray, L_sum: np.ndarray, R_sum: np.ndarray, deg: np.ndarray):
    s = pairs[:,0]; t = pairs[:,1]
    dist = np.linalg.norm(coords[s] - coords[t], axis=1)
    lr = L_sum[s] * R_sum[t]
    ds, dt = deg[s], deg[t]
    Fp = np.stack([dist, lr, ds, dt], axis=1).astype(np.float32)
    m = Fp.mean(axis=0, keepdims=True); sd = Fp.std(axis=0, keepdims=True) + 1e-6
    return (Fp - m) / sd

# ---------- Training ----------

def train_moe(node_feats, coords, X_csr, lig_idx, rec_idx, edge_index_knn, edge_index_lr, edge_index_comb,
              train_pairs, train_labels, val_pairs, val_labels,
              hidden_dim=128, num_layers=2, heads=4, dropout=0.2, epochs=200, lr=1e-3, weight_decay=1e-4,
              patience=10, early_stopping=True, cal_platt=True, tune_threshold=True):
    n = node_feats.shape[0]
    # Precompute sums and edge_attr
    L_sum = np.array(X_csr[:, lig_idx].sum(axis=1)).ravel() if lig_idx.size>0 else np.zeros(n, dtype=np.float32)
    R_sum = np.array(X_csr[:, rec_idx].sum(axis=1)).ravel() if rec_idx.size>0 else np.zeros(n, dtype=np.float32)
    edge_attr_comb = build_edge_attr(edge_index_comb, coords, L_sum, R_sum)
    edge_attr_lr = build_edge_attr(edge_index_lr, coords, L_sum, R_sum) if edge_index_lr.size else np.zeros((0,2), dtype=np.float32)
    deg_comb = build_degrees(n, edge_index_comb)

    # Build models (independent encoders)
    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    ei_knn = torch.tensor(edge_index_knn, dtype=torch.long, device=device)
    ei_lr = torch.tensor(edge_index_lr, dtype=torch.long, device=device)
    ei_comb = torch.tensor(edge_index_comb, dtype=torch.long, device=device)
    ea_comb = torch.tensor(edge_attr_comb, dtype=torch.float32, device=device)
    ea_lr = torch.tensor(edge_attr_lr, dtype=torch.float32, device=device)

    expA = SAGEEncoder(x.shape[1], hidden_dim, num_layers, dropout).to(device)
    expB = GATEncoder(x.shape[1], hidden_dim, num_layers, heads, dropout, concat=False).to(device)
    expC = TransEncoder(x.shape[1], hidden_dim, num_layers, heads, dropout, concat=False).to(device)
    expD = SAGEEncoder(x.shape[1], hidden_dim, num_layers, dropout).to(device)

    clfA = PairClassifier(expA.embed_dim, dropout).to(device)
    clfB = PairClassifier(expB.embed_dim, dropout).to(device)
    clfC = PairClassifier(expC.embed_dim, dropout).to(device)
    clfD = PairClassifier(expD.embed_dim, dropout).to(device)

    gate = GatingNet(in_dim=4, num_experts=4).to(device)

    params = list(expA.parameters())+list(expB.parameters())+list(expC.parameters())+list(expD.parameters()) \
             +list(clfA.parameters())+list(clfB.parameters())+list(clfC.parameters())+list(clfD.parameters())+list(gate.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    tp = torch.tensor(train_pairs, dtype=torch.long, device=device)
    tl = torch.tensor(train_labels, dtype=torch.float32, device=device)
    vp = torch.tensor(val_pairs, dtype=torch.long, device=device)
    vl = torch.tensor(val_labels, dtype=torch.float32, device=device)

    best_aupr, best_state, wait = -1.0, None, 0

    for epoch in range(1, epochs+1):
        expA.train(); expB.train(); expC.train(); expD.train(); gate.train()
        opt.zero_grad()
        hA = expA(x, ei_knn)
        hB = expB(x, ei_comb)
        hC = expC(x, ei_comb, ea_comb)
        hD = expD(x, ei_lr if ei_lr.numel()>0 else ei_knn)
        logA = clfA(hA, tp); logB = clfB(hB, tp); logC = clfC(hC, tp); logD = clfD(hD, tp)
        gfeat = torch.tensor(gating_features(train_pairs, coords, L_sum, R_sum, deg_comb), dtype=torch.float32, device=device)
        gw = gate(gfeat)
        fused_logit = gw[:,0]*logA + gw[:,1]*logB + gw[:,2]*logC + gw[:,3]*logD
        loss = loss_fn(fused_logit, tl)
        loss.backward(); opt.step()

        # Eval
        expA.eval(); expB.eval(); expC.eval(); expD.eval(); gate.eval()
        with torch.no_grad():
            hA = expA(x, ei_knn); hB = expB(x, ei_comb); hC = expC(x, ei_comb, ea_comb); hD = expD(x, ei_lr if ei_lr.numel()>0 else ei_knn)
            logA = clfA(hA, vp); logB = clfB(hB, vp); logC = clfC(hC, vp); logD = clfD(hD, vp)
            gfeat_v = torch.tensor(gating_features(val_pairs, coords, L_sum, R_sum, deg_comb), dtype=torch.float32, device=device)
            gw_v = gate(gfeat_v)
            fused_v = gw_v[:,0]*logA + gw_v[:,1]*logB + gw_v[:,2]*logC + gw_v[:,3]*logD
            vs = torch.sigmoid(fused_v).detach().cpu().numpy()
            vpred = (vs >= 0.5).astype(int)
            auroc = roc_auc_score(val_labels, vs); aupr = average_precision_score(val_labels, vs)
            f1 = f1_score(val_labels, vpred); acc = accuracy_score(val_labels, vpred)
        print(f"[Epoch {epoch:03d}] loss={loss.item():.4f} AUROC={auroc:.4f} AUPR={aupr:.4f} F1={f1:.4f} ACC={acc:.4f}")
        if aupr > best_aupr:
            best_aupr = aupr; wait = 0
            best_state = {"A":expA.state_dict(),"B":expB.state_dict(),"C":expC.state_dict(),"D":expD.state_dict(),
                          "clfA":clfA.state_dict(),"clfB":clfB.state_dict(),"clfC":clfC.state_dict(),"clfD":clfD.state_dict(),
                          "gate":gate.state_dict()}
        else:
            if early_stopping:
                wait += 1
                if wait >= patience:
                    print(f"[INFO] Early stopping at epoch {epoch}")
                    break

    if best_state is not None:
        expA.load_state_dict(best_state["A"]); expB.load_state_dict(best_state["B"]); expC.load_state_dict(best_state["C"]); expD.load_state_dict(best_state["D"])
        clfA.load_state_dict(best_state["clfA"]); clfB.load_state_dict(best_state["clfB"]); clfC.load_state_dict(best_state["clfC"]); clfD.load_state_dict(best_state["clfD"])
        gate.load_state_dict(best_state["gate"])

    # Return components
    return (expA,expB,expC,expD), (clfA,clfB,clfC,clfD), gate, (edge_index_knn, edge_index_lr, edge_index_comb), (edge_attr_lr, edge_attr_comb), (L_sum,R_sum,deg_comb)

# ---------- Prediction ----------

def predict_moe(exps, clfs, gate, x, coords, pairs, ei_knn, ei_lr, ei_comb, ea_lr, ea_comb, L_sum, R_sum, deg_comb):
    expA,expB,expC,expD = exps; clfA,clfB,clfC,clfD = clfs
    tp = torch.tensor(pairs, dtype=torch.long, device=device)
    hA = expA(x, ei_knn)
    hB = expB(x, ei_comb)
    hC = expC(x, ei_comb, ea_comb)
    hD = expD(x, ei_lr if ei_lr.numel()>0 else ei_knn)
    logA = clfA(hA, tp); logB = clfB(hB, tp); logC = clfC(hC, tp); logD = clfD(hD, tp)
    gfeat = torch.tensor(gating_features(pairs, coords, L_sum, R_sum, deg_comb), dtype=torch.float32, device=device)
    gw = gate(gfeat)
    fused = gw[:,0]*logA + gw[:,1]*logB + gw[:,2]*logC + gw[:,3]*logD
    return torch.sigmoid(fused).detach().cpu().numpy()

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="MOE GNN Trainer (4 Experts)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--lr_top_m", type=int, default=8)
    parser.add_argument("--lr_min_strength", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no_early_stopping", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--tune_threshold", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    t0 = time.time()
    adata, X_csr, coords = load_adata(H5AD_FILE)
    lig_idx, rec_idx = load_lr_gene_sets(LR_FILE, list(adata.var_names))
    node_feats = build_node_features(X_csr, coords.astype(np.float32), lig_idx, rec_idx)

    A_knn = build_knn_adjacency(coords.astype(np.float32), k=args.k)
    ei_knn = adj_to_edge_index(A_knn)
    ei_lr = build_lr_edges_simple(X_csr, lig_idx, rec_idx, top_m=args.lr_top_m, min_strength=args.lr_min_strength, symmetric=True)
    edges_set = set(map(tuple, ei_knn.T))
    edges_set.update(map(tuple, ei_lr.T))
    ei_comb = np.array(list(edges_set), dtype=np.int64).T if edges_set else ei_knn

    train_df = read_edges(TRAIN_EDGES, require_label=True)
    val_df = read_edges(VAL_EDGES, require_label=True)
    test_df = read_edges(TEST_EDGES, require_label=False)
    train_pairs = train_df[['source','target']].values.astype(np.int64)
    val_pairs = val_df[['source','target']].values.astype(np.int64)
    test_pairs = test_df[['source','target']].values.astype(np.int64)
    train_labels = train_df['label'].values.astype(np.int64)
    val_labels = val_df['label'].values.astype(np.int64)

    exps, clfs, gate, (E_knn,E_lr,E_comb), (EA_lr,EA_comb), (L_sum,R_sum,deg_comb) = train_moe(
        node_feats, coords, X_csr, lig_idx, rec_idx, ei_knn, ei_lr, ei_comb,
        train_pairs, train_labels, val_pairs, val_labels,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers, heads=args.heads, dropout=args.dropout,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience,
        early_stopping=not args.no_early_stopping, cal_platt=args.calibrate, tune_threshold=args.tune_threshold)

    x = torch.tensor(node_feats, dtype=torch.float32, device=device)
    EI_knn = torch.tensor(E_knn, dtype=torch.long, device=device)
    EI_lr = torch.tensor(E_lr, dtype=torch.long, device=device)
    EI_comb = torch.tensor(E_comb, dtype=torch.long, device=device)
    EA_lr_t = torch.tensor(EA_lr, dtype=torch.float32, device=device)
    EA_comb_t = torch.tensor(EA_comb, dtype=torch.float32, device=device)

    # Validation predictions
    val_scores = predict_moe(exps, clfs, gate, x, coords, val_pairs, EI_knn, EI_lr, EI_comb, EA_lr_t, EA_comb_t, L_sum, R_sum, deg_comb)

    # Optional calibration (Platt scaling)
    calibrator = None
    if args.calibrate:
        lr_cal = LogisticRegression(max_iter=1000)
        lr_cal.fit(val_scores.reshape(-1,1), val_labels)
        def cal_fn(s):
            return lr_cal.predict_proba(s.reshape(-1,1))[:,1]
        calibrator = cal_fn
        val_scores = cal_fn(val_scores)

    # Threshold tuning
    thr = args.threshold
    if args.tune_threshold:
        thresholds = np.linspace(0.0, 1.0, 201)
        accs = [accuracy_score(val_labels, (val_scores >= t).astype(int)) for t in thresholds]
        thr = float(thresholds[int(np.argmax(accs))])
        print(f"[INFO] Tuned threshold={thr:.3f} (val ACC={max(accs):.4f})")

    val_pred = (val_scores >= thr).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(val_labels, val_scores)),
        "aupr": float(average_precision_score(val_labels, val_scores)),
        "f1": float(f1_score(val_labels, val_pred)),
        "accuracy": float(accuracy_score(val_labels, val_pred)),
        "threshold": thr,
        "calibrated": bool(args.calibrate)
    }
    with open(os.path.join(OUT_DIR, "val_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    val_out = val_df.copy(); val_out['score']=val_scores; val_out['pred']=val_pred
    val_out.to_csv(os.path.join(OUT_DIR, "val_predictions.csv"), index=False)

    # Test predictions
    test_scores = predict_moe(exps, clfs, gate, x, coords, test_pairs, EI_knn, EI_lr, EI_comb, EA_lr_t, EA_comb_t, L_sum, R_sum, deg_comb)
    if calibrator is not None:
        test_scores = calibrator(test_scores)
    test_out = test_df.copy(); test_out['score']=test_scores; test_out['pred']=(test_scores>=thr).astype(int)
    test_out.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    submission = test_df[['source','target']].copy(); submission['label']=(test_scores>=thr).astype(int)
    submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

    print(f"[INFO] Done. Metrics: {json.dumps(metrics)}")

if __name__ == "__main__":
    main()