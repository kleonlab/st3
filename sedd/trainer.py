"""
Training utilities for SEDD.

Provides a clean training loop for masked gene expression prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import json
from tqdm import tqdm

from .graph import Graph, AbsorbingGraph
from .noise import NoiseSchedule

Tensor = torch.Tensor


class SEDDTrainer:
    """Trainer for Score-Entropy Discrete Diffusion models.

    Handles the training loop, loss computation, and checkpointing.
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
    ):
        """
        Args:
            model: SEDD model
            graph: Transition graph (e.g., AbsorbingGraph)
            noise: Noise schedule (e.g., LogLinearNoise)
            optimizer: Optional optimizer (default: AdamW)
            scheduler: Optional learning rate scheduler
            device: Device for computation
            gradient_clip: Gradient clipping value
        """
        self.model = model
        self.graph = graph
        self.noise = noise
        self.gradient_clip = gradient_clip

        # Setup device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        self.scheduler = scheduler

        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}

    def compute_loss(
        self,
        x_clean: Tensor,
        mask_ratio: float = 0.15,
    ) -> Tensor:
        """Compute SEDD training loss.

        Process:
        1. Sample random time t ~ Uniform(0, 1)
        2. Compute noise level sigma = noise.total(t)
        3. Apply forward diffusion: x_noised = graph.sample_transition(x_clean, sigma)
        4. Predict score and compute cross-entropy at masked positions

        Args:
            x_clean: Clean data [batch_size, seq_len]
            mask_ratio: Fraction of positions to mask (for absorbing graph)

        Returns:
            Scalar loss value
        """
        batch_size, seq_len = x_clean.shape
        device = x_clean.device

        # Sample random time
        t = torch.rand(batch_size, device=device)
        sigma = self.noise.total(t)

        # Apply forward diffusion
        if isinstance(self.graph, AbsorbingGraph):
            # For absorbing: use mask_ratio to control masking
            x_noised = self._mask_tokens(x_clean, mask_ratio, sigma)
        else:
            x_noised = self.graph.sample_transition(x_clean, sigma)

        # Get model prediction
        pred_score = self.model.score(x_noised, sigma)

        # Compute loss at masked positions
        mask_idx = getattr(self.graph, 'mask_index', self.graph.num_states - 1)
        is_masked = (x_noised == mask_idx)

        if not is_masked.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cross-entropy loss at masked positions
        pred_at_mask = pred_score[is_masked]
        target_at_mask = x_clean[is_masked]

        loss = F.cross_entropy(pred_at_mask, target_at_mask)

        return loss

    def _mask_tokens(
        self,
        x: Tensor,
        mask_ratio: float,
        sigma: Tensor,
    ) -> Tensor:
        """Mask tokens using absorbing graph with time-dependent probability.

        Args:
            x: Clean tokens [batch_size, seq_len]
            mask_ratio: Base mask ratio
            sigma: Noise level [batch_size]

        Returns:
            Masked tokens
        """
        batch_size, seq_len = x.shape
        device = x.device
        mask_idx = self.graph.mask_index

        # Time-dependent mask probability
        # At t=0 (sigma=0), mask less; at t=1 (sigma=high), mask more
        # Use sigma to modulate the mask probability
        p_mask = 1 - torch.exp(-sigma)  # [batch_size]
        p_mask = p_mask.view(-1, 1)  # [batch_size, 1]

        # Sample mask
        mask = torch.rand(batch_size, seq_len, device=device) < p_mask

        # Apply mask
        x_masked = x.clone()
        x_masked[mask] = mask_idx

        return x_masked

    def train_step(self, batch: Tensor, mask_ratio: float = 0.15) -> float:
        """Single training step.

        Args:
            batch: Batch of clean data [batch_size, seq_len]
            mask_ratio: Mask ratio for training

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        batch = batch.to(self.device)
        loss = self.compute_loss(batch, mask_ratio)

        loss.backward()

        # Gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )

        self.optimizer.step()
        self.step += 1

        return loss.item()

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        mask_ratio: float = 0.15,
    ) -> float:
        """Validate on a validation set.

        Args:
            val_loader: Validation data loader
            mask_ratio: Mask ratio for validation

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(self.device)
            loss = self.compute_loss(batch, mask_ratio)
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
        callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            mask_ratio: Fraction of positions to mask
            log_interval: Steps between logging
            val_interval: Epochs between validation
            checkpoint_dir: Directory to save checkpoints
            callback: Optional callback function(trainer, epoch, metrics)

        Returns:
            Training history
        """
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training loop
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in pbar:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                loss = self.train_step(batch, mask_ratio)
                epoch_loss += loss
                num_batches += 1

                if self.step % log_interval == 0:
                    pbar.set_postfix({"loss": f"{loss:.4f}"})

            avg_train_loss = epoch_loss / max(num_batches, 1)
            self.history["train_loss"].append(avg_train_loss)

            # Validation
            val_loss = None
            if val_loader and (epoch + 1) % val_interval == 0:
                val_loss = self.validate(val_loader, mask_ratio)
                self.history["val_loss"].append(val_loss)

                # Update learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    if checkpoint_dir:
                        self.save_checkpoint(checkpoint_dir / "best.pt")

            # Logging
            log_msg = f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.4f}"
            print(log_msg)

            # Callback
            if callback:
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                }
                callback(self, epoch, metrics)

            # Periodic checkpoint
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1}.pt")

        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir / "final.pt")

        return self.history

    def save_checkpoint(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
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
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state
        """
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


def create_trainer(
    model: nn.Module,
    graph: Graph,
    noise: NoiseSchedule,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    device: Optional[torch.device] = None,
) -> SEDDTrainer:
    """Convenience function to create a trainer with standard settings.

    Args:
        model: SEDD model
        graph: Transition graph
        noise: Noise schedule
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps for scheduler
        device: Device for computation

    Returns:
        Configured SEDDTrainer
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )

    # Cosine annealing with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return SEDDTrainer(
        model=model,
        graph=graph,
        noise=noise,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
