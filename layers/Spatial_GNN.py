"""
Pipeline:
  1. Stack target + neighbor time series → (B, N_total, L, n_vars)
  2. Flatten (L, n_vars) → L*n_vars as node feature
  3. Linear encode → (B, N_total, d_node)
  4. GCN layer (normalized adjacency + self-loops) → (B, N_total, d_node)
  5. Global average pooling over nodes → (B, d_node)
  6. Linear projection → E_spatial: (B, d_out)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialGNN(nn.Module):
    """
    Args:
        n_vars  : number of input features per node (same as enc_in)
        d_node  : hidden dimension inside GNN (= d_model)
        d_out   : output embedding dimension (= d_model, to match other branches)
        dropout : dropout rate
    """

    def __init__(self, n_vars: int, seq_len: int, d_node: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        # Step 3: encode raw node features to d_node
        # node feature = flattened time series: (L, n_vars) → L*n_vars
        self.node_encoder = nn.Linear(seq_len * n_vars, d_node)
        self.seq_len = seq_len
        self.n_vars = n_vars
        # Step 4: GCN weight matrix W
        self.gcn_weight = nn.Linear(d_node, d_node, bias=False)
        # Step 6: project to d_out
        self.output_proj = nn.Linear(d_node, d_out)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.GELU()

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetric normalisation with self-loops:
            Â = D^{-1/2} (A + I) D^{-1/2}
        adj: (N, N) binary
        """
        N = adj.shape[0]
        A = adj + torch.eye(N, device=adj.device)   # add self-loops
        deg = A.sum(dim=1)                            # degree vector (N,)
        d_inv_sqrt = deg.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.0
        D = torch.diag(d_inv_sqrt)                    # (N, N)
        return D @ A @ D                              # (N, N)

    def forward(
        self,
        x_target: torch.Tensor,
        neighbor_x_encs: list,
        subgraph_adj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_target        : (B, L, n_vars)  target BA time series
            neighbor_x_encs : list of N tensors (B, L, n_vars), one per neighbor
                              Empty list → isolated node, only target is used
            subgraph_adj    : (N_total, N_total) binary adjacency of subgraph
                              (target node is always at index 0)
        Returns:
            E_spatial: (B, d_out)
        """
        # 1. Stack all nodes: target first, then neighbors
        all_nodes = [x_target] + neighbor_x_encs          # N_total tensors
        X = torch.stack(all_nodes, dim=1)                  # (B, N_total, L, n_vars)

        # 2. Treat T as part of node feature: flatten (L, n_vars) → L*n_vars
        B, N_total, L, _ = X.shape
        X = X.reshape(B, N_total, L * self.n_vars)         # (B, N_total, L*n_vars)

        # 3. Linear encode node features
        X = self.act(self.node_encoder(X))                 # (B, N_total, d_node)

        # 4. GCN: H = σ(Â X W)
        A_hat = self._normalize_adj(subgraph_adj)          # (N_total, N_total)
        # broadcast Â across batch: (1, N, N) @ (B, N, d) → (B, N, d)
        AX = A_hat.unsqueeze(0) @ X                        # (B, N_total, d_node)
        H = self.act(self.gcn_weight(AX))                  # (B, N_total, d_node)
        H = self.dropout(H)

        # 5. Global average pooling over nodes
        z = H.mean(dim=1)                                   # (B, d_node)

        # 6. Project to d_out and normalise
        E_spatial = self.norm(self.output_proj(z))         # (B, d_out)
        return E_spatial
