"""Trainer for cell-type conditioned perturbation prediction using discrete diffusion.

This trainer handles the cell-type conditioned perturbation prediction task:
- Input: perturbed cell (noised) + perturbation label + cell-type label
- Output: predicted perturbed cell (denoised)

Training procedure:
1. Sample diffusion time t
2. Apply discrete diffusion (masking) to the perturbed cell
3. Model predicts the perturbed cell from: masked_perturbed + perturbation_label + celltype_label
4. Loss: cross-entropy at masked positions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import json
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sedd.graph import Graph, AbsorbingGraph
from sedd.noise import NoiseSchedule

Tensor = torch.Tensor


class CellTypePerturbationTrainer:
    """Trainer for cell-type conditioned perturbation prediction using discrete diffusion.

    This trainer handles the cell-type conditioned perturbation prediction task:
    - Input: perturbed cell (noised) + perturbation label + cell-type label
    - Output: predicted perturbed cell

    Training procedure:
    1. Sample diffusion time t
    2. Apply discrete diffusion (masking) to the perturbed cell
    3. Model predicts the perturbed cell from: masked_perturbed + pert_label + celltype_label
    4. Loss: cross-entropy at masked positions
    """

    def __init__(
        self,
        model: nn.Module,
        graph: Graph,
        noise: NoiseSchedule,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = None,
        gradient_clip: float = 1.0,
        cond_label_lookup: Optional[Tensor] = None,
        celltype_label_lookup: Optional[Tensor] = None,
    ):
        """
        Args:
            model: The cell-type conditioned perturbation model
            graph: Diffusion graph (e.g., AbsorbingGraph)
            noise: Noise schedule (e.g., LogLinearNoise)
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            device: Device to train on
            gradient_clip: Gradient clipping value
            cond_label_lookup: Optional tensor for perturbation label lookup
            celltype_label_lookup: Optional tensor for cell-type label lookup
        """
        self.model = model
        self.graph = graph
        self.noise = noise
        self.gradient_clip = gradient_clip
        self.cond_label_lookup = cond_label_lookup
        self.celltype_label_lookup = celltype_label_lookup

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Move lookups to device if provided
        if self.cond_label_lookup is not None:
            self.cond_label_lookup = self.cond_label_lookup.to(self.device)
            print(f"Using conditional label lookup with shape: {self.cond_label_lookup.shape}")

        if self.celltype_label_lookup is not None:
            self.celltype_label_lookup = self.celltype_label_lookup.to(self.device)
            print(f"Using cell-type label lookup with shape: {self.celltype_label_lookup.shape}")

        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        self.scheduler = scheduler
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def compute_loss(
        self,
        pert_labels: Tensor,
        celltype_labels: Tensor,
        perturbed: Tensor,
        mask_ratio: float = 0.15,
    ) -> Tensor:
        """
        Compute perturbation prediction loss with cell-type conditioning.

        Args:
            pert_labels: Perturbation labels [batch]
            celltype_labels: Cell-type labels [batch]
            perturbed: True perturbed expression [batch, seq_len]
            mask_ratio: Masking ratio (used by absorbing graph)

        Returns:
            Cross-entropy loss at masked positions
        """
        batch_size, seq_len = perturbed.shape
        device = perturbed.device

        # Sample diffusion time
        t = torch.rand(batch_size, device=device)
        sigma = self.noise.total(t)

        # Apply discrete diffusion (masking) to perturbed cells
        if isinstance(self.graph, AbsorbingGraph):
            x_noised = self._mask_tokens(perturbed, mask_ratio, sigma)
        else:
            x_noised = self.graph.sample_transition(perturbed, sigma)

        # Model predicts perturbed from noised + perturbation label + cell-type label
        loss = self.model.get_loss(
            x_perturbed=perturbed,
            x_noised=x_noised,
            sigma=sigma,
            pert_labels=pert_labels,
            celltype_labels=celltype_labels,
            graph=self.graph,
        )

        return loss

    def _mask_tokens(
        self,
        x: Tensor,
        mask_ratio: float,
        sigma: Tensor,
    ) -> Tensor:
        """Apply masking to tokens based on diffusion time."""
        batch_size, seq_len = x.shape
        device = x.device
        mask_idx = self.graph.mask_index

        # Masking probability based on diffusion time
        p_mask = 1 - torch.exp(-sigma)  # [batch_size]
        p_mask = p_mask.view(-1, 1)  # [batch_size, 1]

        # Sample mask positions
        mask = torch.rand(batch_size, seq_len, device=device) < p_mask

        # Apply masking
        x_masked = x.clone()
        x_masked[mask] = mask_idx

        return x_masked

    def train_step(
        self,
        batch: Tuple[Tensor, ...],
        mask_ratio: float = 0.15
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Unpack batch - handle cell-load dictionary format
        if isinstance(batch, dict):
            # Cell-load batch format with cell-type
            perturbed = batch['pert_cell_emb'].to(self.device)
            pert_emb = batch['pert_emb'].to(self.device)

            # Get cell-type labels
            if 'cell_type_emb' in batch:
                celltype_emb = batch['cell_type_emb'].to(self.device)
                # Convert one-hot to indices if needed
                if celltype_emb.dim() == 2 and celltype_emb.shape[1] > 1:
                    celltype_labels = celltype_emb.argmax(dim=-1)
                else:
                    celltype_labels = celltype_emb.squeeze(-1).long()
            elif 'cell_type' in batch:
                celltype_labels = batch['cell_type'].to(self.device).long()
            else:
                raise ValueError("Cell-type labels not found in batch. Expected 'cell_type_emb' or 'cell_type' key.")

            # Convert one-hot perturbation embeddings to indices
            if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                pert_labels = pert_emb.argmax(dim=-1)
            else:
                pert_labels = pert_emb.squeeze(-1).long()
        else:
            # Legacy tuple format: (pert_labels, celltype_labels, perturbed)
            if len(batch) == 3:
                pert_labels, celltype_labels, perturbed = batch
            else:
                raise ValueError(
                    f"Expected batch with 3 elements (pert_labels, celltype_labels, perturbed), "
                    f"got {len(batch)} elements"
                )
            pert_labels = pert_labels.to(self.device)
            celltype_labels = celltype_labels.to(self.device)
            perturbed = perturbed.to(self.device)

        pert_labels = self._normalize_pert_labels(pert_labels)
        celltype_labels = self._normalize_celltype_labels(celltype_labels)

        # Apply conditional label lookups if provided
        pert_labels = self._apply_cond_label_lookup(pert_labels)
        celltype_labels = self._apply_celltype_label_lookup(celltype_labels)

        # Ensure labels are long type (handle both scalar and vector labels)
        if pert_labels.dim() == 1:
            pert_labels = pert_labels.long()
        if celltype_labels.dim() == 1:
            celltype_labels = celltype_labels.long()

        # Round and convert to long for discrete tokens
        perturbed = torch.round(perturbed).long()

        # Compute loss
        loss = self.compute_loss(pert_labels, celltype_labels, perturbed, mask_ratio)

        # Backward pass
        loss.backward()

        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

        self.optimizer.step()
        self.step += 1

        return loss.item()

    def _apply_cond_label_lookup(self, pert_labels: Tensor) -> Tensor:
        """Replace perturbation labels with conditional labels from lookup."""
        if self.cond_label_lookup is None:
            return pert_labels

        cond_labels = self.cond_label_lookup[pert_labels]

        # For scalar labels, fall back to original labels when missing or out of range
        if cond_labels.dim() == 1:
            num_perturbations = self.model.num_perturbations
            invalid = (cond_labels < 0) | (cond_labels >= num_perturbations)
            if invalid.any():
                invalid_count = invalid.sum().item()
                print(
                    f"WARNING: {invalid_count} conditional labels out of range; "
                    "falling back to original perturbation indices for those samples."
                )
                cond_labels = cond_labels.clone()
                cond_labels[invalid] = pert_labels[invalid]

        return cond_labels

    def _apply_celltype_label_lookup(self, celltype_labels: Tensor) -> Tensor:
        """Replace cell-type labels with conditional labels from lookup."""
        if self.celltype_label_lookup is None:
            return celltype_labels

        ct_labels = self.celltype_label_lookup[celltype_labels]

        # For scalar labels, fall back to original labels when missing or out of range
        if ct_labels.dim() == 1:
            num_cell_types = self.model.num_cell_types
            invalid = (ct_labels < 0) | (ct_labels >= num_cell_types)
            if invalid.any():
                invalid_count = invalid.sum().item()
                print(
                    f"WARNING: {invalid_count} cell-type labels out of range; "
                    "falling back to original cell-type indices for those samples."
                )
                ct_labels = ct_labels.clone()
                ct_labels[invalid] = celltype_labels[invalid]

        return ct_labels

    def _normalize_pert_labels(self, pert_labels: Tensor) -> Tensor:
        """Ensure perturbation labels are in [0, num_perturbations - 1]."""
        pert_labels = pert_labels.long()
        num_perturbations = self.model.num_perturbations

        if pert_labels.numel() == 0:
            return pert_labels

        min_label = int(pert_labels.min().item())
        max_label = int(pert_labels.max().item())

        # Common case: labels are 1-based but embedding expects 0-based
        if min_label == 1 and max_label == num_perturbations:
            pert_labels = pert_labels - 1
            return pert_labels

        if min_label < 0 or max_label >= num_perturbations:
            raise ValueError(
                "Perturbation labels out of range for embedding. "
                f"Expected [0, {num_perturbations - 1}], got [{min_label}, {max_label}]. "
                "Ensure dataset labels are zero-based or match num_perturbations."
            )

        return pert_labels

    def _normalize_celltype_labels(self, celltype_labels: Tensor) -> Tensor:
        """Ensure cell-type labels are in [0, num_cell_types - 1]."""
        celltype_labels = celltype_labels.long()
        num_cell_types = self.model.num_cell_types

        if celltype_labels.numel() == 0:
            return celltype_labels

        min_label = int(celltype_labels.min().item())
        max_label = int(celltype_labels.max().item())

        # Common case: labels are 1-based but embedding expects 0-based
        if min_label == 1 and max_label == num_cell_types:
            celltype_labels = celltype_labels - 1
            return celltype_labels

        if min_label < 0 or max_label >= num_cell_types:
            raise ValueError(
                "Cell-type labels out of range for embedding. "
                f"Expected [0, {num_cell_types - 1}], got [{min_label}, {max_label}]. "
                "Ensure dataset labels are zero-based or match num_cell_types."
            )

        return celltype_labels

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        mask_ratio: float = 0.15,
    ) -> float:
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            # Unpack batch - handle cell-load dictionary format
            if isinstance(batch, dict):
                perturbed = batch['pert_cell_emb'].to(self.device)
                pert_emb = batch['pert_emb'].to(self.device)

                # Get cell-type labels
                if 'cell_type_emb' in batch:
                    celltype_emb = batch['cell_type_emb'].to(self.device)
                    if celltype_emb.dim() == 2 and celltype_emb.shape[1] > 1:
                        celltype_labels = celltype_emb.argmax(dim=-1)
                    else:
                        celltype_labels = celltype_emb.squeeze(-1).long()
                elif 'cell_type' in batch:
                    celltype_labels = batch['cell_type'].to(self.device).long()
                else:
                    raise ValueError("Cell-type labels not found in batch")

                # Convert one-hot perturbation embeddings to indices
                if pert_emb.dim() == 2 and pert_emb.shape[1] > 1:
                    pert_labels = pert_emb.argmax(dim=-1)
                else:
                    pert_labels = pert_emb.squeeze(-1).long()
            else:
                # Legacy tuple format
                pert_labels, celltype_labels, perturbed = batch
                pert_labels = pert_labels.to(self.device)
                celltype_labels = celltype_labels.to(self.device)
                perturbed = perturbed.to(self.device)

            pert_labels = self._normalize_pert_labels(pert_labels)
            celltype_labels = self._normalize_celltype_labels(celltype_labels)

            # Apply conditional label lookups if provided
            pert_labels = self._apply_cond_label_lookup(pert_labels)
            celltype_labels = self._apply_celltype_label_lookup(celltype_labels)

            # Ensure labels are long type
            if pert_labels.dim() == 1:
                pert_labels = pert_labels.long()
            if celltype_labels.dim() == 1:
                celltype_labels = celltype_labels.long()

            # Round and convert to long for discrete tokens
            perturbed = torch.round(perturbed).long()

            loss = self.compute_loss(pert_labels, celltype_labels, perturbed, mask_ratio)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 100,
        mask_ratio: float = 0.15,
        log_interval: int = 100,
        val_interval: int = 1,
        checkpoint_dir: Optional[str] = None,
        save_interval: int = 10,
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:

        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume from last epoch if we have history
        start_epoch = self.epoch + 1 if self.epoch > 0 else 0

        for epoch in range(start_epoch, num_epochs):
            self.epoch = epoch

            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                loss = self.train_step(batch, mask_ratio)
                epoch_loss += loss
                num_batches += 1

                if self.step % log_interval == 0:
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            val_loss = None
            if val_loader and (epoch + 1) % val_interval == 0:
                val_loss = self.validate(val_loader, mask_ratio)
                self.history["val_loss"].append(val_loss)

                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / "best.pt")

            log_msg = f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.4f}"
            print(log_msg)

            if callback:
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }
                callback(self, epoch, metrics)

            if checkpoint_dir and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / "final.pt")

        return self.history

    def save_checkpoint(self, path: str):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True):
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
