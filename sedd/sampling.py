"""
Sampling strategies for SEDD.

Implements reverse diffusion sampling to generate/impute gene expression.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Callable
from tqdm import tqdm

from .graph import Graph, AbsorbingGraph
from .noise import NoiseSchedule

Tensor = torch.Tensor


class Sampler(ABC):
    """Abstract base class for SEDD samplers."""

    def __init__(
        self,
        model,
        graph: Graph,
        noise: NoiseSchedule,
        num_steps: int = 100,
        device: torch.device = None,
    ):
        """
        Args:
            model: SEDD model with .score() method
            graph: Transition graph
            noise: Noise schedule
            num_steps: Number of sampling steps
            device: Device for computation
        """
        self.model = model
        self.graph = graph
        self.noise = noise
        self.num_steps = num_steps
        self.device = device or next(model.parameters()).device

    @abstractmethod
    def step(
        self,
        x: Tensor,
        t: float,
        dt: float,
    ) -> Tensor:
        """Single reverse diffusion step.

        Args:
            x: Current state [batch, seq_len]
            t: Current time (1 = fully noised, 0 = clean)
            dt: Time step size (negative, going backwards)

        Returns:
            Updated state [batch, seq_len]
        """
        pass

    @torch.no_grad()
    def sample(
        self,
        x_init: Tensor,
        mask_positions: Optional[Tensor] = None,
        show_progress: bool = True,
    ) -> Tensor:
        """Run full reverse diffusion sampling.

        Args:
            x_init: Initial state (usually all masked or partially masked)
            mask_positions: Boolean tensor indicating which positions to impute
            show_progress: Whether to show progress bar

        Returns:
            Generated/imputed samples [batch, seq_len]
        """
        self.model.eval()
        x = x_init.clone().to(self.device)

        # Time goes from 1 (fully noised) to 0 (clean)
        times = torch.linspace(1, 0, self.num_steps + 1, device=self.device)
        dt = -1.0 / self.num_steps

        iterator = range(self.num_steps)
        if show_progress:
            iterator = tqdm(iterator, desc="Sampling")

        for i in iterator:
            t = times[i]
            x = self.step(x, t, dt)

            # Optionally keep non-masked positions fixed
            if mask_positions is not None:
                x = torch.where(mask_positions, x, x_init)

        # Final denoising step
        x = self.denoise(x)

        return x

    @torch.no_grad()
    def denoise(self, x: Tensor) -> Tensor:
        """Final denoising step: pick most likely token at each position.

        Args:
            x: Current state [batch, seq_len]

        Returns:
            Denoised state [batch, seq_len]
        """
        # Get score at very low noise level
        sigma = torch.tensor([0.01], device=self.device)
        score = self.model.score(x, sigma)

        # For masked positions, pick the most likely non-mask token
        mask_idx = self.graph.mask_index if hasattr(self.graph, 'mask_index') else -1
        is_masked = (x == mask_idx)

        if is_masked.any():
            # Get probabilities (exclude mask token)
            probs = F.softmax(score[..., :-1], dim=-1)
            # Sample or argmax
            sampled = probs.argmax(dim=-1)
            x = torch.where(is_masked, sampled, x)

        return x


class EulerSampler(Sampler):
    """Euler method for reverse diffusion.

    Uses the score to compute a reverse rate and takes a step.
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

    def step(self, x: Tensor, t: float, dt: float) -> Tensor:
        """Euler step for reverse diffusion.

        Computes: x_{t+dt} = x_t + rate * dt + noise
        """
        t_tensor = torch.tensor([t], device=self.device)
        sigma = self.noise.total(t_tensor)
        dsigma = self.noise.rate(t_tensor) * (-dt)  # dt is negative

        # Get score from model
        score = self.model.score(x, sigma)  # [batch, seq, vocab]

        # For absorbing diffusion, compute reverse rate
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
        """Euler step for absorbing diffusion."""
        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        # Only update masked positions
        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        # Compute transition probabilities from score
        # score is log p(clean | noised), use it as reverse rate
        probs = F.softmax(score / self.temperature, dim=-1)

        # Probability of staying masked vs transitioning
        # p_unmask ~ dsigma * exp(-sigma) * score
        p_stay = torch.exp(-dsigma)

        # Sample whether to unmask
        unmask_prob = (1 - p_stay) * is_masked.float()
        do_unmask = torch.rand_like(unmask_prob) < unmask_prob

        # For positions that unmask, sample new token
        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        # Don't pick mask token
        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        # Apply updates
        x_new = torch.where(do_unmask, new_tokens, x)

        return x_new

    def _euler_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        """General Euler step using score for transition rates."""
        # Compute transition probabilities
        probs = F.softmax(score / self.temperature, dim=-1)

        # Sample from probabilities with some staying probability
        p_stay = torch.exp(-dsigma).item()

        # With probability p_stay, keep current token
        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        # Sample new tokens
        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)


class AnalyticSampler(Sampler):
    """Analytic sampler using closed-form transition probabilities.

    More accurate than Euler but requires analytic solutions.
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

    def step(self, x: Tensor, t: float, dt: float) -> Tensor:
        """Analytic step using staggered score."""
        t_tensor = torch.tensor([t], device=self.device)
        t_next = torch.tensor([t + dt], device=self.device)

        sigma = self.noise.total(t_tensor)
        sigma_next = self.noise.total(t_next)
        dsigma = sigma - sigma_next

        # Get score at current noise level
        score = self.model.score(x, sigma)

        # Compute reverse transition probabilities
        if isinstance(self.graph, AbsorbingGraph):
            return self._analytic_step_absorbing(x, score, sigma, dsigma)
        else:
            return self._analytic_step_general(x, score, sigma, dsigma)

    def _analytic_step_absorbing(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        """Analytic step for absorbing diffusion."""
        batch_size, seq_len = x.shape
        mask_idx = self.graph.mask_index

        is_masked = (x == mask_idx)

        if not is_masked.any():
            return x

        # Staggered score: approximate p_{sigma - dsigma}(z) / p_sigma(x)
        # For absorbing: this gives us the probability of each clean token
        staggered = score * torch.exp(-dsigma)
        probs = F.softmax(staggered / self.temperature, dim=-1)

        # Compute reverse transition probabilities
        # P(unmask to token j | masked) ~ p(token j was original) * exp(-dsigma)
        p_stay_masked = torch.exp(-dsigma).item()

        # Sample new states
        # First decide: stay masked or unmask?
        do_unmask = torch.rand_like(is_masked.float()) > p_stay_masked

        # For unmasking, sample from staggered score
        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(batch_size, seq_len)

        # Don't pick mask token during unmasking
        new_tokens = new_tokens.clamp(max=mask_idx - 1)

        # Apply: unmask if both is_masked and do_unmask
        should_unmask = is_masked & do_unmask
        x_new = torch.where(should_unmask, new_tokens, x)

        return x_new

    def _analytic_step_general(
        self,
        x: Tensor,
        score: Tensor,
        sigma: Tensor,
        dsigma: Tensor,
    ) -> Tensor:
        """General analytic step."""
        # Similar to Euler but with better transition probability estimates
        staggered = score * torch.exp(-dsigma)
        probs = F.softmax(staggered / self.temperature, dim=-1)

        p_stay = torch.exp(-dsigma).item()
        stay_mask = torch.rand(x.shape, device=x.device) < p_stay

        new_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(x.shape)

        return torch.where(stay_mask, x, new_tokens)


def get_sampler(
    name: str,
    model,
    graph: Graph,
    noise: NoiseSchedule,
    **kwargs
) -> Sampler:
    """Factory function for samplers.

    Args:
        name: "euler" or "analytic"
        model: SEDD model
        graph: Transition graph
        noise: Noise schedule
        **kwargs: Additional arguments

    Returns:
        Sampler instance
    """
    samplers = {
        "euler": EulerSampler,
        "analytic": AnalyticSampler,
    }

    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Choose from {list(samplers.keys())}")

    return samplers[name](model, graph, noise, **kwargs)


@torch.no_grad()
def impute_masked(
    model,
    graph: Graph,
    noise: NoiseSchedule,
    x: Tensor,
    mask: Tensor,
    sampler: str = "euler",
    num_steps: int = 100,
    temperature: float = 1.0,
    show_progress: bool = True,
) -> Tensor:
    """Convenience function to impute masked positions.

    Args:
        model: SEDD model
        graph: Transition graph
        noise: Noise schedule
        x: Input data with some positions to mask
        mask: Boolean tensor where True = impute this position
        sampler: Sampler type ("euler" or "analytic")
        num_steps: Number of sampling steps
        temperature: Sampling temperature
        show_progress: Show progress bar

    Returns:
        Data with masked positions imputed
    """
    device = next(model.parameters()).device
    x = x.clone().to(device)
    mask = mask.to(device)

    # Set masked positions to mask token
    mask_idx = graph.mask_index if hasattr(graph, 'mask_index') else graph.num_states - 1
    x[mask] = mask_idx

    # Create sampler and run
    s = get_sampler(
        sampler,
        model,
        graph,
        noise,
        num_steps=num_steps,
        temperature=temperature,
        device=device,
    )

    # Sample with fixed non-masked positions
    result = s.sample(x, mask_positions=mask, show_progress=show_progress)

    return result
