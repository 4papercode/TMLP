"""
Precompute spatial metadata for ST-TMLP:
  1. Pearson correlation matrix (28x28) from historical demand
  2. First-order neighbor lists per BA (from WECC adjacency matrix)

Outputs (saved to utils/):
  wecc_similarity.npy   -- (28, 28) float64 correlation matrix
  wecc_neighbors.json   -- {ba_name: [neighbor_indices], ...}
  wecc_ba_names.json    -- ordered list of 28 BA names (index reference)
"""

import os
import json
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "..", "cleaned_data_16-24")
ADJ_PATH     = os.path.join(PROJECT_ROOT, "..", "WECC Physical Connecions", "WECC_adjacent.csv")
OUT_DIR      = SCRIPT_DIR  # save alongside this script

# ── Load adjacency matrix ─────────────────────────────────────────────────────
adj_df = pd.read_csv(ADJ_PATH, index_col=0)

# The adjacency matrix uses "LADWP"; data files use "LDWP" — normalise to data names
adj_df.index   = [n.replace("LADWP", "LDWP") for n in adj_df.index]
adj_df.columns = [n.replace("LADWP", "LDWP") for n in adj_df.columns]

BA_NAMES = adj_df.columns.tolist()   # 28 names, canonical order
adj_np   = adj_df.values.astype(int) # (28, 28) binary

print(f"BA list ({len(BA_NAMES)}): {BA_NAMES}")

# ── Load demand time series for all BAs ───────────────────────────────────────
demand_cols = []
for ba in BA_NAMES:
    fpath = os.path.join(DATA_DIR, f"{ba}_cleaned_historical_data.csv")
    df = pd.read_csv(fpath, usecols=["Cleaned_Demand_MWh"])
    demand_cols.append(df["Cleaned_Demand_MWh"].values)
    print(f"  Loaded {ba}: {len(df)} rows")

demand_matrix = np.stack(demand_cols, axis=1)   # (T, 28)
print(f"\nDemand matrix shape: {demand_matrix.shape}")

# Check for NaNs
nan_counts = np.isnan(demand_matrix).sum(axis=0)
for i, ba in enumerate(BA_NAMES):
    if nan_counts[i] > 0:
        print(f"  WARNING: {ba} has {nan_counts[i]} NaN values — will be filled with column mean")
        col = demand_matrix[:, i]
        col_mean = np.nanmean(col)
        demand_matrix[:, i] = np.where(np.isnan(col), col_mean, col)

# ── Compute Pearson correlation matrix ────────────────────────────────────────
# Use training split only (first 70% of data) to avoid data leakage
T = demand_matrix.shape[0]
train_end = int(T * 0.7)
S = np.corrcoef(demand_matrix[:train_end].T)   # (28, 28)

print(f"\nSimilarity matrix (Pearson, train split):")
print(f"  min={S.min():.4f}  max={S.max():.4f}  mean={S.mean():.4f}")
print(f"  diagonal all 1s: {np.allclose(np.diag(S), 1.0)}")

# ── Build first-order neighbor lists ─────────────────────────────────────────
# For each BA i: neighbors = indices j where adj[i,j] == 1 (excludes self)
neighbors_dict = {}
for i, ba in enumerate(BA_NAMES):
    nb_indices = np.where(adj_np[i, :] > 0)[0].tolist()
    neighbors_dict[ba] = nb_indices

    # Report + flag isolated nodes
    if len(nb_indices) == 0:
        print(f"  {ba}: isolated node (no neighbors) — will use self-only")
    else:
        nb_names = [BA_NAMES[j] for j in nb_indices]
        sim_vals = [round(float(S[i, j]), 4) for j in nb_indices]
        print(f"  {ba}: {len(nb_indices)} neighbors {nb_names}")
        print(f"       similarities: {sim_vals}")

# ── Save outputs ──────────────────────────────────────────────────────────────
sim_path  = os.path.join(OUT_DIR, "wecc_similarity.npy")
nb_path   = os.path.join(OUT_DIR, "wecc_neighbors.json")
name_path = os.path.join(OUT_DIR, "wecc_ba_names.json")

np.save(sim_path, S)
json.dump(neighbors_dict, open(nb_path,   "w"), indent=2)
json.dump(BA_NAMES,       open(name_path, "w"), indent=2)

print(f"\nSaved:")
print(f"  {sim_path}")
print(f"  {nb_path}")
print(f"  {name_path}")
