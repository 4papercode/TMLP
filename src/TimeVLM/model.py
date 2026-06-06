import os
import sys
import numpy as np
import torch
import torch.nn as nn
import einops
from PIL import Image

# Import custom modules, assuming they are stored in the parent directory
sys.path.append("../")
from src.TimeVLM.vlm_manager import VLMManager
from layers.Embed import PatchEmbedding
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.Spatial_GNN import SpatialGNN
from cnn import PI_FeatureExtractor
from layers.models_mae import *
from transformers.models.vilt import *

def mmd_loss(x, y, sigma=1.0):
    """
    Compute MMD (Maximum Mean Discrepancy) between two sets of features.
    Args:
        x, y: (B, n_vars, pred_len) tensors
        sigma: Gaussian kernel bandwidth
    Returns:
        scalar MMD loss
    """
    x = x.reshape(x.size(0), -1)  # (B, D)
    y = y.reshape(y.size(0), -1)  # (B, D)

    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    K_xx = torch.exp(-(rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
    K_yy = torch.exp(-(ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
    K_xy = torch.exp(-(rx.t() + ry - 2 * xy) / (2 * sigma ** 2))

    return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()


class PatchMemoryBank:
    def __init__(self, max_size, patch_size, feature_dim, device=None):
        """
        Initialize the patch memory bank.
        
        Args:
            max_size (int): Maximum number of patches to store.
            patch_size (int): Size of each patch.
            feature_dim (int): Dimensionality of each patch feature.
            device (torch.device): Device to store memory bank on (CPU/GPU).
        """
        self.max_size = max_size
        self.patch_size = patch_size
        self.feature_dim = feature_dim
        self.device = device if device is not None else torch.device('cpu')
        self.patches = torch.zeros((max_size, feature_dim), device=self.device)  # [100, d_model]
        self.ptr = 0

    def update(self, new_patches):
        """
        Update the patch memory bank with new patches using circular buffer strategy.
        
        Args:
            new_patches (Tensor): New patches to add to the memory bank.
        """
        n = new_patches.size(0)
        new_patches_flat = new_patches.mean(dim=1)  # [n, d_model]
        
        if self.ptr + n > self.max_size:
            # Wrap around if the memory bank is full
            remaining_space = self.max_size - self.ptr
            self.patches[self.ptr:] = new_patches_flat[:remaining_space]        
            remaining_patches = n - remaining_space
            if remaining_patches >= self.max_size:
                self.patches[:] = new_patches_flat[-self.max_size:]
                self.ptr = 0
            else:
                self.patches[:remaining_patches] = new_patches_flat[remaining_space:]
                self.ptr = remaining_patches
        else:
            self.patches[self.ptr:self.ptr + n] = new_patches_flat
            self.ptr += n

    def retrieve(self, query_patches, top_k=5):
        """
        Retrieve the top-k most similar patches from the memory bank.
        
        Args:
            query_patches (Tensor): Query patches for retrieval.
            top_k (int): Number of nearest neighbors to retrieve.
        
        Returns:
            retrieved_patches (Tensor): Retrieved patches from the memory bank.
            indices (Tensor): Indices of the retrieved patches.
        """
        query_flat = query_patches.mean(dim=1)  # [224, d_model]
        memory_flat = self.patches  # [100, d_model]
        
        similarity = torch.matmul(query_flat, memory_flat.T)  # [224, 100]
        _, indices = similarity.topk(top_k, dim=-1)
        
        retrieved_patches = self.patches[indices]
        return retrieved_patches, indices


class Model(nn.Module):
    """
    Time-VLM model with image and text modalities for enhanced time series forecasting.
    """
    def __init__(self, config, **kwargs):
        super(Model, self).__init__()
        self.config = config
        self.vlm_manager = VLMManager(config)
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.config.gpu))
        else:
            self.device = torch.device('cpu')
        self.use_mem_gate = config.use_mem_gate
        
        # Initialize patch memory bank
        self.patch_memory_bank = PatchMemoryBank(
            max_size=config.patch_memory_size,  # e.g., 100 patches
            patch_size=config.patch_len,
            feature_dim=config.d_model,
            device=self.device
        )
        
        self._init_modules(config)
        self.vlm_model = self.vlm_manager.model

    def _init_modules(self, config):
        self.patch_embedding = PatchEmbedding(
            config.d_model, 
            config.patch_len, 
            config.stride, 
            config.padding, 
            config.dropout
        )
        self.head_nf = config.d_model * int((config.seq_len - config.patch_len) / config.stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        
        # Main memory prediction head
        self.memory_head = nn.Sequential(
            nn.Linear(self.head_nf, config.pred_len),
            nn.Dropout(config.dropout)
        )
        
        # Main temporal head
        self.temporal_head = nn.Sequential(
            nn.Linear(self.head_nf, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        self.multimodal_head = nn.Sequential(
            nn.Linear(config.d_model, config.pred_len),
            nn.LayerNorm(config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Multimodal enhancement
        self.multimodal_enhancement = nn.Sequential(
            nn.Linear(self.vlm_manager.hidden_size * 2, config.d_model),  # Combine vision and text
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Cross-modal attention for feature enhancement
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Memory fusion gate
        if self.use_mem_gate:
            self.memory_fusion_gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 2),
                nn.Softmax(dim=-1)
            )

        # Prediction fusion gate
        self.gate = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Linear(config.pred_len, 2),
            nn.Softmax(dim=-1)
        )
        
        # Final fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.pred_len * 2, config.pred_len),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Memory-related modules
        self.local_memory_mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model)
        )
        
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.learnable_image_module = LearnableTimeSeriesToImage(
            input_dim=3, 
            hidden_dim=48, 
            output_channels=3 if config.three_channel_image else 1,
            image_size=config.image_size, 
            periodicity=config.periodicity
        )
        
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter
        self.layer_norm = nn.LayerNorm(config.d_model)

        # Spatial-Augmented Learner (SAL) — disabled when use_spatial=False
        self.use_spatial = getattr(config, 'use_spatial', False)
        if self.use_spatial:
            self.spatial_gnn = SpatialGNN(
                n_vars=config.enc_in,
                seq_len=config.seq_len,
                d_node=config.d_model,
                d_out=config.d_model,
                dropout=config.dropout,
            )
            self.spatial_head = nn.Sequential(
                nn.Linear(config.d_model, config.pred_len),
                nn.Dropout(config.dropout),
            )

        # Topology-Augmented Learner (TAL) — disabled when use_topology=False
        self.use_topology = getattr(config, 'use_topology', False)
        if self.use_topology:
            self.topo_extractor_1 = PI_FeatureExtractor(feature_dim=config.d_model)
            self.topo_extractor_2 = PI_FeatureExtractor(feature_dim=config.d_model)

            self.topo_self_attention = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=4,
                dropout=config.dropout,
                batch_first=True
            )

            self.topo_token_norm = nn.LayerNorm(config.d_model)
            self.topo_fusion_norm = nn.LayerNorm(config.d_model)

            self.topo_head = nn.Sequential(
                nn.Linear(config.d_model, config.pred_len),
                nn.Dropout(config.dropout),
            )

        # Hierarchical gate:
        # Full model (SAL+TAL): RAL+VAL→F_mm, SAL+TAL→F_GT, F_mm+F_GT→F_tem
        # Partial (SAL only or TAL only): flat 3-way gate
        if self.use_spatial and self.use_topology:
            # Level-1 gate: SAL + TAL → F_GT
            self.gt_gate = nn.Sequential(
                nn.Linear(config.pred_len * 2, config.pred_len),
                nn.GELU(),
                nn.Linear(config.pred_len, 2),
                nn.Softmax(dim=-1),
            )
            # Level-2 gate: F_mm + F_GT → F_tem
            self.final_gate = nn.Sequential(
                nn.Linear(config.pred_len * 2, config.pred_len),
                nn.GELU(),
                nn.Linear(config.pred_len, 2),
                nn.Softmax(dim=-1),
            )
        elif self.use_spatial or self.use_topology:
            # Flat 3-way gate: RAL + VAL + (SAL or TAL)
            self.flat_gate = nn.Sequential(
                nn.Linear(config.pred_len * 3, config.pred_len),
                nn.GELU(),
                nn.Linear(config.pred_len, 3),
                nn.Softmax(dim=-1),
            )

    def _compute_local_memory(self, patches):
        """Compute local memory by retrieving and fusing similar patches"""
        # Retrieve similar patches from memory bank
        retrieved_patches, _ = self.patch_memory_bank.retrieve(patches, top_k=self.config.top_k)
        
        # Process retrieved patches with local MLP
        local_memory = self.local_memory_mlp(retrieved_patches)
        
        # Average over retrieved patches
        local_memory = local_memory.mean(dim=1, keepdim=True)
        
        # Residual connection with original patches
        local_memory = local_memory + patches
        
        return local_memory

    def _compute_global_memory(self, patches):
        """Compute global memory by aggregating information across all patches"""
        # Self-attention to capture global dependencies
        attn_output, _ = self.memory_attention(
            query=patches,
            key=patches,
            value=patches
        )
        
        # Update patch memory bank with current patches
        self.patch_memory_bank.update(patches.detach())
        
        if self.use_mem_gate:
            return attn_output  # Return full attention output for advanced gating
        else:
            # Return global context for simple gating (original behavior)
            return attn_output.mean(dim=1, keepdim=True)

    def forward_prediction(self, x_enc, vision_embeddings, text_embeddings,
                           neighbor_x_encs=None, subgraph_adj=None,
                           pi_images_1=None, pi_images_2=None):
        B, L, n_vars = x_enc.shape

        # 1. Process temporal features
        patches, _ = self.patch_embedding(x_enc.transpose(1, 2))  # [B * n_vars, n_patches, d_model]
        
        # 2. Compute local and global memory
        local_memory = self._compute_local_memory(patches)  # [B * n_vars, n_patches, d_model]
        global_memory = self._compute_global_memory(patches)  # [B * n_vars, n_patches, d_model] or [B * n_vars, 1, d_model]
        
        # 3. Combine local and global memory
        if self.use_mem_gate:
            # Advanced memory fusion with gating
            combined_features = torch.cat([local_memory, global_memory], dim=-1)  # [B * n_vars, n_patches, d_model*2]
            gate_weights = self.memory_fusion_gate(combined_features)  # [B * n_vars, n_patches, 2]
            
            # Weighted fusion
            memory_features = (
                gate_weights[:, :, 0:1] * local_memory +
                gate_weights[:, :, 1:2] * global_memory
            )  # [B * n_vars, n_patches, d_model]
        else:
            # Simple addition (original behavior)
            memory_features = local_memory + global_memory  # [B * n_vars, n_patches, d_model]

        # 4. Get temporal predictions
        memory_features = self.flatten(memory_features)  # [B * n_vars, head_nf]
        temporal_features = self.temporal_head(memory_features)  # [B, n_vars, d_model]
        memory_features = self.memory_head(memory_features)  # [B * n_vars, pred_len]
        temporal_features = einops.rearrange(temporal_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, d_model]
        memory_features = einops.rearrange(memory_features, '(b n) d -> b n d', b=B, n=n_vars)  # [B, n_vars, pred_len]
        
        # 5. Process multimodal features
        multimodal_features = torch.cat([vision_embeddings, text_embeddings], dim=-1)  # [B, hidden_size * 2]
        multimodal_features = self.multimodal_enhancement(multimodal_features)  # [B, d_model]
        multimodal_features = multimodal_features.unsqueeze(1).expand(-1, n_vars, -1)  # [B, n_vars, d_model]
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        
        # 6. Cross-modal attention enhancement
        temporal_features = temporal_features / torch.norm(temporal_features, dim=-1, keepdim=True)
        multimodal_features = multimodal_features / torch.norm(multimodal_features, dim=-1, keepdim=True)
        multimodal_features, _ = self.cross_attention(
            query=temporal_features,
            key=multimodal_features,
            value=multimodal_features
        )  # [B, n_vars, d_model]
        
        # 7. Normalize cross attention output
        multimodal_features = self.layer_norm(multimodal_features)    # [B, n_vars, d_model]
        multimodal_features = self.multimodal_head(multimodal_features)  # [B, n_vars, pred_len]
        
        # 8. SAL + TAL branches + hierarchical gating
        use_sal = (self.use_spatial
                   and neighbor_x_encs is not None
                   and len(neighbor_x_encs) > 0
                   and subgraph_adj is not None)
        use_tal = (self.use_topology and pi_images_1 is not None and pi_images_2 is not None)

        if use_sal:
            spatial_emb = self.spatial_gnn(
                x_enc, neighbor_x_encs, subgraph_adj.to(x_enc.device)
            )  # (B, d_model)
            spatial_emb = spatial_emb.unsqueeze(1).expand(-1, n_vars, -1)  # (B, n_vars, d_model)
            spatial_pred = self.spatial_head(spatial_emb)                  # (B, n_vars, pred_len)

        if use_tal:
            pi_images_1 = pi_images_1.float().to(x_enc.device)
            pi_images_2 = pi_images_2.float().to(x_enc.device)

            topo_emb_1 = self.topo_extractor_1(pi_images_1)              # (B, d_model)
            topo_emb_2 = self.topo_extractor_2(pi_images_2)              # (B, d_model)
            topo_tokens = torch.stack([topo_emb_1, topo_emb_2],dim=1)
            topo_tokens = self.topo_token_norm(topo_tokens)
            
            fused_tokens, _ = self.topo_self_attention(
                query=topo_tokens,
                key=topo_tokens,
                value=topo_tokens
            )
            topo_emb = fused_tokens.mean(dim=1)
            topo_emb = self.topo_fusion_norm(topo_emb)

            topo_emb = topo_emb.unsqueeze(1).expand(-1, n_vars, -1)       # (B, n_vars, d_model)
            topo_pred = self.topo_head(topo_emb)                           # (B, n_vars, pred_len)

        if use_sal and use_tal:
            # Hierarchical fusion:
            # Step 1: F_mm = gate(RAL, VAL)
            w_mm = self.gate(torch.cat([memory_features, multimodal_features], dim=-1))
            f_mm = w_mm[:, :, 0:1] * memory_features + w_mm[:, :, 1:2] * multimodal_features

            # Step 2: F_GT = gate(SAL, TAL)
            w_gt = self.gt_gate(torch.cat([spatial_pred, topo_pred], dim=-1))
            f_gt = w_gt[:, :, 0:1] * spatial_pred + w_gt[:, :, 1:2] * topo_pred

            # Step 3: F_tem = gate(F_mm, F_GT)
            w_final = self.final_gate(torch.cat([f_mm, f_gt], dim=-1))
            fused_features = w_final[:, :, 0:1] * f_mm + w_final[:, :, 1:2] * f_gt

        elif use_sal or use_tal:
            # Flat 3-way gate: RAL + VAL + (SAL or TAL)
            extra = spatial_pred if use_sal else topo_pred
            w = self.flat_gate(torch.cat([memory_features, multimodal_features, extra], dim=-1))
            fused_features = (w[:, :, 0:1] * memory_features +
                              w[:, :, 1:2] * multimodal_features +
                              w[:, :, 2:3] * extra)

        else:
            # Baseline: 2-way gate RAL + VAL
            w = self.gate(torch.cat([memory_features, multimodal_features], dim=-1))
            fused_features = w[:, :, 0:1] * memory_features + w[:, :, 1:2] * multimodal_features

        # 9. Final fusion
        predictions = self.fusion_layer(
            torch.cat([memory_features, fused_features], dim=-1)
        ) + memory_features  # (B, n_vars, pred_len)

        # Return intermediate features for alignment loss (only when full model)
        if use_sal and use_tal:
            return predictions.permute(0, 2, 1), f_mm, f_gt, memory_features
        else:
            return predictions.permute(0, 2, 1), None, None, memory_features

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None,
                neighbor_x_encs=None, subgraph_adj=None,
                pi_images_1=None, pi_images_2=None):
        B, L, D = x_enc.shape
        x_enc = x_enc.to(self.device)

        # Move neighbor tensors to the same device
        if neighbor_x_encs is not None:
            neighbor_x_encs = [nb.float().to(self.device) for nb in neighbor_x_encs]

        # Normalize input
        x_enc, means, stdev = self._normalize_input(x_enc)

        # Convert time series data to images and generate text prompts
        images = self.vision_augmented_learner(x_enc, self.config.image_size, self.config.seq_len, self.config.periodicity)
        prompts = self.text_augmented_learner(x_enc, self.config.content, self.config.pred_len, self.config.seq_len)

        # Process inputs with the VLM
        vision_embeddings, text_embeddings = self.vlm_manager.process_inputs(B, images, prompts)

        # Main prediction branch (RAL + VAL + SAL + TAL)
        predictions, f_mm, f_gt, memory_features = self.forward_prediction(
            x_enc, vision_embeddings, text_embeddings,
            neighbor_x_encs=neighbor_x_encs,
            subgraph_adj=subgraph_adj,
            pi_images_1=pi_images_1,
            pi_images_2=pi_images_2,
        )

        # Compute alignment losses (MMD) — only for full model (SAL+TAL present)
        zero = torch.tensor(0.0, device=x_enc.device)
        if f_gt is not None and f_mm is not None:
            align_gt = mmd_loss(f_gt,  memory_features)
            align_mm = mmd_loss(f_mm,  memory_features)
        else:
            align_gt = zero
            align_mm = zero

        # Denormalize output
        y = self._denormalize_output(predictions, means, stdev)
        return y, align_gt, align_mm

    def _normalize_input(self, x):
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x = x / stdev
        return x, means, stdev

    def _denormalize_output(self, y, means, stdev):
        y = y * (stdev.repeat(1, self.config.pred_len, 1))
        y = y + (means.repeat(1, self.config.pred_len, 1))
        return y

    def text_augmented_learner(self, x_enc, description, pred_len, seq_len, top_k=5):
        """
        Generate text prompts for the language model based on time series data.
        Each variable in the time series will have its own prompt.
        """
        B, T, n_vars = x_enc.shape  # Get batch size, sequence length, and number of variables

        # Initialize a list to store prompts for each batch
        prompts = []
    
        # Calculate overall statistics for each batch
        for b in range(B):
            # Calculate statistics for the current batch
            min_value = torch.min(x_enc[b]).item()  # Overall minimum value for the batch
            max_value = torch.max(x_enc[b]).item()  # Overall maximum value for the batch
            median_value = torch.median(x_enc[b]).item()  # Overall median value for the batch
            trend = x_enc[b].diff(dim=0).sum().item()  # Overall trend for the batch

            # Determine the overall trend direction
            trend_direction = "upward" if trend > 0 else "downward"
                
            prompt_parts = [
                "The time series is converted into an image using 1D and 2D convolutional layers, highlighting trends, periodic patterns, and multi-scale features for forecasting.",
                f"Dataset: {description}",
                f"Task: Forecast the next {pred_len} steps using the past {seq_len} steps.",
                f"Input statistics: min value = {min_value:.3f}, max value = {max_value:.3f}, median value = {median_value:.3f}, the overall trend is {trend_direction}."
            ]
            prompt = " ".join(prompt_parts)
            prompt = prompt[:self.vlm_manager.max_input_text_length] if len(prompt) > self.vlm_manager.max_input_text_length else prompt
            prompts.append(prompt)  

        return prompts

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        """
        Convert time series data into 3-channel image tensors.
        """
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:            
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        
        # Normalize images to [0, 255] as uint8
        images = self._normalize_images(images)
        
        # Optionally save images
        if self.config.save_images:
            self.save_images(images)

        return images
    
    @staticmethod
    def _normalize_images(images):
        """
        Normalize image tensors to [0, 255] as uint8.
        Assumes images are in [0, 1] or need to be scaled.
        
        Args:
        - images (Tensor): Input images with shape [B, C, H, W]
        
        Returns:
        - Tensor: Normalized images as uint8 with shape [B, C, H, W]
        """
        # Compute min and max per image across all channels and spatial dimensions
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-5
        scale = (max_vals - min_vals).clamp(min=epsilon)
        # Normalize to [0, 1]
        images = (images - min_vals) / scale
        # Scale to [0, 255] and clamp to ensure valid range
        images = images.clamp(0, 1)
        
        return images

    @torch.no_grad()
    def save_images(self, images):
        """
        Save the generated images.

        Args:
        - images: A tensor containing the images to be saved with shape [B, C, H, W]
        """
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)
        
        for i, img_tensor in enumerate(images):
            # Move to CPU and convert to numpy
            img_tensor = img_tensor.cpu().numpy()
            
            # Check channel count and handle accordingly
            if img_tensor.shape[0] == 3:
                # RGB image: Convert from [C, H, W] to [H, W, C]
                img_tensor = np.transpose(img_tensor, (1, 2, 0))
                mode = 'RGB'
            elif img_tensor.shape[0] == 1:
                # Grayscale image: Convert from [C, H, W] to [H, W]
                img_tensor = np.squeeze(img_tensor, 0)
                mode = 'L'
            else:
                print(f"Warning: Unexpected number of channels {img_tensor.shape[0]} for image {i}. Skipping...")
                continue
            
            # Ensure data type is uint8
            if img_tensor.dtype != np.uint8:
                img_tensor = img_tensor.astype(np.uint8)
            
            # Create PIL image and save
            try:
                img = Image.fromarray(img_tensor, mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue
