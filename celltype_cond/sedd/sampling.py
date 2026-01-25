"""Samplers for cell-type conditioned perturbation prediction.

This module provides sampling algorithms for cell-type conditioned models,
enabling cell-type-specific generation of perturbed cell states.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sedd.graph import Graph, AbsorbingGraph
from sedd.noise import NoiseSchedule
from sedd.sampling import Sampler

Tensor = torch.Tensor


class CellTypePerturbationEulerSampler(Sampler):
    """Euler sampler with perturbation AND cell-type conditioning.

    This sampler is specifically designed for models that require both
    perturbation labels and cell-type labels, such as SEDDCellTypePerturbationTransformer.
    It passes both pert_labels and celltype_labels through the entire denoising chain.
    """

    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
        temperature: float = 1.0,
    ):
        super().__init__(model, graph, noise, num_steps, device)
        self.temperature = temperature

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        pert_labels: Tensor,
        celltype_labels: Tensor,
        mask_positions: Optional[Tensor] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """
        Sample perturbed cells conditioned on perturbation and cell-type labels.

        Args:
            x_init: Initial state (typically all masked) [batch, seq_len]
            pert_labels: Perturbation labels [batch]
            celltype_labels: Cell-type labels [batch]
            mask_positions: Optional positions to keep fixed
            show_progress: Show progress bar

        Returns:
            Sampled sequences [batch, seq_len]
        """
        self.model.eval()
        x = x_init.clone().to(self.device)
        pert_labels = pert_labels.to(self.device)
        celltype_labels = celltype_labels.to(self.device)

        times = torch.linspace(1, 0, self.num_steps + 1, device=self.device)
        dt = -1.0 / self.num_steps

        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = times[i]
            x = self.step(x, t, dt, pert_labels, celltype_labels)

            if mask_positions is not None:
                x = torch.where(mask_positions, x, x_init)

        x = self.denoise(x, pert_labels, celltype_labels)

        return x

    def step(
        self,
        x: Tensor,
        t: float,
        dt: float,
        pert_labels: Tensor,
        celltype_labels: Tensor,
    ) -> Tensor:
        """Single denoising step with perturbation and cell-type conditioning."""
        t_tensor = torch.tensor([t], device=self.device)
        sigma = self.noise.total(t_tensor)
        dsigma = self.noise.rate(t_tensor) * (-dt)

        # Get score WITH perturbation AND cell-type conditioning
        score = self.model.score(x, sigma, pert_labels, celltype_labels)

        if isinstance(self.graph, AbsorbingGraph):
            return self._euler_step_absorbing(x, score, sigma, dsigma)
        else:
            return self._euler_step_general(x, score, sigma, dsigma)

    def _euler_step_absorbing(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        """Absorbing diffusion step."""
        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma)

        unmask_prob = (1 - p_stay) * is_masked.float()
        do_unmask = torch.rand_like(unmask_prob) < unmask_prob

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        x_new = torch.where(do_unmask, new_tokens, x)

        return x_new

    def _euler_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        """General diffusion step."""
        probs = F.softmax(score / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma).item()

        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)

    @torch.no_grad()
    def denoise(
        self,
        x: Tensor,
        pert_labels: Tensor,
        celltype_labels: Tensor,
    ) -> Tensor:
        """Final denoising step with perturbation and cell-type conditioning."""
        sigma = torch.tensor([0.01], device=self.device)
        score = self.model.score(x, sigma, pert_labels, celltype_labels)

        mask_idx = self.graph.mask_index if hasattr(self.graph, 'mask_index') else -1
        is_masked = (x == mask_idx)

        if is_masked.any():
            probs = F.softmax(score[..., :-1], dim=-1)
            sampled = probs.argmax(dim=-1)
            x = torch.where(is_masked, sampled, x)

        return x


def get_celltype_sampler(
    name: str,
    model,
    graph: Graph,
    noise: NoiseSchedule,
    **kwargs
) -> CellTypePerturbationEulerSampler:
    """Get a cell-type conditioned sampler by name.

    Args:
        name: Sampler name (currently only 'euler' supported)
        model: Cell-type conditioned perturbation model
        graph: Diffusion graph
        noise: Noise schedule
        **kwargs: Additional sampler arguments

    Returns:
        Sampler instance
    """
    samplers = {
        "euler": CellTypePerturbationEulerSampler,
    }

    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")

    return samplers[name](model, graph, noise, **kwargs)
