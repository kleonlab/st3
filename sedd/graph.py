"""
Graph structures for discrete diffusion.

Defines the forward transition dynamics through rate matrices.
The rate matrix Q defines instantaneous transition probabilities,
and exp(sigma * Q) gives the transition matrix over time sigma.
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple

Tensor = torch.Tensor


class Graph(ABC):
    """Abstract base class for transition graphs.

    The graph defines how tokens transition during the forward diffusion.
    Key concept: rate matrix Q where Q[i,j] = rate of transitioning from i to j.
    """

    def __init__(self, num_states: int):
        """
        Args:
            num_states: Number of discrete states (vocabulary size + mask token)
        """
        self.num_states = num_states

    @abstractmethod
    def rate(self, x: Tensor) -> Tensor:
        """Compute rate matrix columns for given states.

        Args:
            x: Current states [batch_size, seq_len]

        Returns:
            Rate vectors [batch_size, seq_len, num_states]
            where result[b, l, j] = rate of x[b,l] -> j
        """
        pass

    @abstractmethod
    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Compute transition probabilities after time sigma.

        Args:
            x: Current states [batch_size, seq_len]
            sigma: Noise level [batch_size] or scalar

        Returns:
            Transition probabilities [batch_size, seq_len, num_states]
        """
        pass

    @abstractmethod
    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Sample from transition distribution.

        Args:
            x: Current states [batch_size, seq_len]
            sigma: Noise level [batch_size] or scalar

        Returns:
            Sampled states [batch_size, seq_len]
        """
        pass

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """Sample from the limiting distribution (t -> infinity).

        Args:
            shape: Output shape
            device: Device to create tensor on

        Returns:
            Sampled states
        """
        pass


class AbsorbingGraph(Graph):
    """Absorbing diffusion graph.

    All states eventually transition to an absorbing "mask" state.
    The mask state is the last index (num_states - 1).

    Rate matrix Q:
        Q[i, mask] = 1 for all i != mask (transition to mask)
        Q[i, j] = 0 for j != mask
        Q[mask, mask] = 0 (mask is absorbing)

    This is ideal for masked prediction tasks like masked RNA-seq.
    """

    def __init__(self, num_states: int):
        """
        Args:
            num_states: Total states including mask token (last index)
        """
        super().__init__(num_states)
        self.mask_index = num_states - 1

    def rate(self, x: Tensor) -> Tensor:
        """Rate of transitioning from x to each state.

        Only transition to mask state has non-zero rate.
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Rate is 1 for transitioning to mask, 0 otherwise
        rate = torch.zeros(batch_size, seq_len, self.num_states, device=device)

        # All non-mask states transition to mask with rate 1
        # Mask states stay (rate 0 to self, but we handle this in transition)
        is_not_mask = (x != self.mask_index).float()
        rate[..., self.mask_index] = is_not_mask

        return rate

    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Transition probabilities after time sigma.

        For absorbing: P(stay) = exp(-sigma), P(mask) = 1 - exp(-sigma)
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)

        batch_size, seq_len = x.shape
        device = x.device

        # Probability of staying in current state
        p_stay = torch.exp(-sigma)  # [batch, 1, 1]

        # Build transition probabilities
        probs = torch.zeros(batch_size, seq_len, self.num_states, device=device)

        # Set probability of staying in current state
        probs.scatter_(2, x.unsqueeze(-1), p_stay.expand(batch_size, seq_len, 1))

        # Probability of transitioning to mask
        is_not_mask = (x != self.mask_index).float().unsqueeze(-1)
        p_to_mask = (1 - p_stay) * is_not_mask
        probs[..., self.mask_index:self.mask_index + 1] += p_to_mask

        # For tokens already masked, keep them masked
        is_mask = (x == self.mask_index).float().unsqueeze(-1)
        probs[..., self.mask_index:self.mask_index + 1] += is_mask * (1 - p_stay)

        return probs

    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Sample from transition distribution."""
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)

        device = x.device
        batch_size, seq_len = x.shape

        # Probability of staying
        p_stay = torch.exp(-sigma)  # [batch, 1]

        # Sample whether to transition
        uniform = torch.rand(batch_size, seq_len, device=device)
        transition_to_mask = uniform > p_stay

        # Apply transition
        result = x.clone()
        result[transition_to_mask] = self.mask_index

        return result

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """Limiting distribution is all mask tokens."""
        return torch.full(shape, self.mask_index, dtype=torch.long, device=device)


class UniformGraph(Graph):
    """Uniform diffusion graph.

    All states can transition to any other state with equal probability.

    Rate matrix Q:
        Q[i, j] = 1/(num_states - 1) for i != j
        Q[i, i] = -1 (to ensure rows sum to 0)

    This provides more mixing but is less interpretable for masking tasks.
    """

    def __init__(self, num_states: int):
        super().__init__(num_states)

    def rate(self, x: Tensor) -> Tensor:
        """Rate of transitioning from x to each state."""
        batch_size, seq_len = x.shape
        device = x.device

        # Uniform rate to all other states
        rate_val = 1.0 / (self.num_states - 1)
        rate = torch.full(
            (batch_size, seq_len, self.num_states),
            rate_val,
            device=device
        )

        # Zero rate to stay in current state (handled separately)
        rate.scatter_(2, x.unsqueeze(-1), 0.0)

        return rate

    def transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Transition probabilities for uniform diffusion.

        For uniform: P(stay) = 1/S + (1 - 1/S) * exp(-sigma * S / (S-1))
                     P(other) = 1/S - 1/S * exp(-sigma * S / (S-1))
        """
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)

        batch_size, seq_len = x.shape
        device = x.device
        S = self.num_states

        # Eigenvalue decay
        decay = torch.exp(-sigma * S / (S - 1))

        # Probability of each state (uniform baseline)
        p_uniform = 1.0 / S

        # Off-diagonal: approach uniform
        p_other = p_uniform * (1 - decay)

        # Diagonal: start at 1, approach uniform
        p_stay = p_uniform + (1 - p_uniform) * decay

        # Build probability matrix
        probs = torch.full(
            (batch_size, seq_len, self.num_states),
            0.0,
            device=device
        )

        # Set uniform probability for all states
        probs = probs + p_other

        # Adjust for staying in current state
        extra_stay = p_stay - p_other
        probs.scatter_add_(2, x.unsqueeze(-1), extra_stay.expand(batch_size, seq_len, 1))

        return probs

    def sample_transition(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Sample from uniform transition distribution."""
        probs = self.transition(x, sigma)
        # Sample from categorical
        return torch.multinomial(
            probs.view(-1, self.num_states),
            num_samples=1
        ).view(x.shape)

    def sample_limiting(self, shape: Tuple[int, ...], device: torch.device) -> Tensor:
        """Limiting distribution is uniform over all states."""
        return torch.randint(0, self.num_states, shape, device=device)


def get_graph(name: str, num_states: int) -> Graph:
    """Factory function for graphs.

    Args:
        name: "absorbing" or "uniform"
        num_states: Number of discrete states

    Returns:
        Graph instance
    """
    graphs = {
        "absorbing": AbsorbingGraph,
        "uniform": UniformGraph,
    }

    if name not in graphs:
        raise ValueError(f"Unknown graph: {name}. Choose from {list(graphs.keys())}")

    return graphs[name](num_states)
