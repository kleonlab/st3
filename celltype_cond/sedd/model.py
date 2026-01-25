"""Cell-type conditioned SEDD Perturbation Transformer.

This module extends SEDDPerturbationTransformer to incorporate cell-type
conditioning, enabling cell-type-specific perturbation predictions.

Architecture:
    - Inherits token, gene, and time embeddings from SEDDTransformer
    - Adds perturbation embedding for perturbation conditioning
    - Adds cell-type embedding for cell-type conditioning
    - Combines time, perturbation, and cell-type embeddings for adaptive layer norm

Training:
    - Input: control cell expression + perturbation label + cell-type label
    - Target: perturbed cell expression for that specific cell type
    - Uses discrete diffusion with masking
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sedd.model import (
    SinusoidalEmbedding,
    RotaryEmbedding,
    MultiHeadAttention,
    FeedForward,
    AdaptiveLayerNorm,
    TransformerBlock,
    SEDDTransformer,
)

Tensor = torch.Tensor


class SEDDCellTypePerturbationTransformer(SEDDTransformer):
    """SEDD Transformer with perturbation AND cell-type conditioning.

    This model extends SEDDTransformer to condition on both perturbation labels
    and cell-type labels, enabling cell-type-specific perturbation predictions.

    Architecture:
        - Inherits token, gene, and time embeddings from SEDDTransformer
        - Adds perturbation embedding for conditioning
        - Adds cell-type embedding for cell-type-specific conditioning
        - Combines time + perturbation + cell-type embeddings for adaptive layer norm

    Training:
        - Input: perturbed cell expression (noised) + perturbation label + cell-type label
        - Target: perturbed cell expression (clean)
        - Uses discrete diffusion with masking
    """

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        num_cell_types: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 4096,
        precomputed_pert_emb_dim: int = None,
        precomputed_celltype_emb_dim: int = None,
    ):
        """
        Args:
            num_genes: Number of genes (sequence length)
            num_bins: Number of expression bins (vocabulary size, excluding mask)
            num_perturbations: Number of unique perturbation conditions
            num_cell_types: Number of unique cell types
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer blocks
            num_heads: Number of attention heads
            ff_mult: Feed-forward expansion factor
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            precomputed_pert_emb_dim: Dimension of pre-computed perturbation embeddings (e.g., 320 for ESM2)
            precomputed_celltype_emb_dim: Dimension of pre-computed cell-type embeddings (if using)
        """
        super().__init__(
            num_genes=num_genes,
            num_bins=num_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        self.num_perturbations = num_perturbations
        self.num_cell_types = num_cell_types
        self.precomputed_pert_emb_dim = precomputed_pert_emb_dim
        self.precomputed_celltype_emb_dim = precomputed_celltype_emb_dim

        # Perturbation embedding
        self.pert_embed = nn.Embedding(num_perturbations, hidden_dim)

        # Pre-computed perturbation embedding projection (e.g., for protein embeddings)
        if precomputed_pert_emb_dim is not None:
            self.precomputed_pert_proj = nn.Linear(precomputed_pert_emb_dim, hidden_dim)
        else:
            self.precomputed_pert_proj = None

        # Perturbation projection MLP
        self.pert_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Cell-type embedding
        self.celltype_embed = nn.Embedding(num_cell_types, hidden_dim)

        # Pre-computed cell-type embedding projection (if using pre-computed embeddings)
        if precomputed_celltype_emb_dim is not None:
            self.precomputed_celltype_proj = nn.Linear(precomputed_celltype_emb_dim, hidden_dim)
        else:
            self.precomputed_celltype_proj = None

        # Cell-type projection MLP
        self.celltype_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self._init_weights()

    def forward(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        celltype_labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Input tokens [batch, seq_len]
            sigma: Diffusion time [batch] or scalar
            pert_labels: Perturbation labels [batch] (indices or embeddings)
            celltype_labels: Cell-type labels [batch] (indices or embeddings)
            mask: Optional attention mask [batch, seq_len]

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(batch_size)

        # Token and position embeddings
        tok_emb = self.token_embed(x)
        pos_idx = torch.arange(seq_len, device=device)
        pos_emb = self.gene_embed(pos_idx).unsqueeze(0)
        h = tok_emb + pos_emb

        # Time embedding
        t_emb = self.time_embed(sigma)

        # Perturbation embedding
        if pert_labels.dim() == 1:
            # pert_labels are indices, use embedding layer
            p_emb = self.pert_embed(pert_labels.long())
        else:
            # pert_labels are already embeddings (from cond_label_lookup)
            if self.precomputed_pert_proj is not None:
                p_emb = self.precomputed_pert_proj(pert_labels)
            else:
                p_emb = pert_labels
        p_emb = self.pert_proj(p_emb)

        # Cell-type embedding
        if celltype_labels.dim() == 1:
            # celltype_labels are indices, use embedding layer
            ct_emb = self.celltype_embed(celltype_labels.long())
        else:
            # celltype_labels are already embeddings
            if self.precomputed_celltype_proj is not None:
                ct_emb = self.precomputed_celltype_proj(celltype_labels)
            else:
                ct_emb = celltype_labels
        ct_emb = self.celltype_proj(ct_emb)

        # Combined conditioning: time + perturbation + cell-type
        cond = t_emb + p_emb + ct_emb

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond, mask)

        # Output
        h = self.out_norm(h, cond)
        logits = self.out_proj(h)

        return logits

    def score(
        self,
        x: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        celltype_labels: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute log probabilities (scores) for each token."""
        logits = self.forward(x, sigma, pert_labels, celltype_labels, mask)
        score = F.log_softmax(logits, dim=-1)
        return score

    def get_loss(
        self,
        x_perturbed: Tensor,
        x_noised: Tensor,
        sigma: Tensor,
        pert_labels: Tensor,
        celltype_labels: Tensor,
        graph,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute perturbation prediction loss with cell-type conditioning.

        Args:
            x_perturbed: True perturbed expression (target) [batch, seq_len]
            x_noised: Noised perturbed expression (input) [batch, seq_len]
            sigma: Diffusion time [batch]
            pert_labels: Perturbation labels [batch]
            celltype_labels: Cell-type labels [batch]
            graph: Diffusion graph (for mask index)
            mask: Optional attention mask

        Returns:
            Cross-entropy loss at masked positions
        """
        # Predict perturbed expression from noised perturbed + perturbation + cell-type
        pred_score = self.score(x_noised, sigma, pert_labels, celltype_labels, mask)

        # Only compute loss at masked positions
        is_masked = (x_noised == self.mask_index)

        if not is_masked.any():
            return torch.tensor(0.0, device=x_perturbed.device, requires_grad=True)

        pred_at_mask = pred_score[is_masked]  # [num_masked, vocab_size]
        target_at_mask = x_perturbed[is_masked]  # [num_masked]

        loss = F.cross_entropy(pred_at_mask, target_at_mask)

        return loss


class SEDDCellTypePerturbationTransformerSmall(SEDDCellTypePerturbationTransformer):
    """Small version of cell-type conditioned perturbation transformer."""

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        num_cell_types: int,
        **kwargs
    ):
        defaults = dict(
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, num_cell_types, **defaults)


class SEDDCellTypePerturbationTransformerMedium(SEDDCellTypePerturbationTransformer):
    """Medium version of cell-type conditioned perturbation transformer."""

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        num_cell_types: int,
        **kwargs
    ):
        defaults = dict(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, num_cell_types, **defaults)


class SEDDCellTypePerturbationTransformerLarge(SEDDCellTypePerturbationTransformer):
    """Large version of cell-type conditioned perturbation transformer."""

    def __init__(
        self,
        num_genes: int,
        num_bins: int,
        num_perturbations: int,
        num_cell_types: int,
        **kwargs
    ):
        defaults = dict(
            hidden_dim=512,
            num_layers=8,
            num_heads=8,
            ff_mult=4.0,
            dropout=0.1,
        )
        defaults.update(kwargs)
        super().__init__(num_genes, num_bins, num_perturbations, num_cell_types, **defaults)
