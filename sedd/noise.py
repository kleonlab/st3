"""
Noise schedules for discrete diffusion.

Provides schedules that control the rate and total amount of noise
applied during the forward diffusion process.
"""

import torch
from abc import ABC, abstractmethod
from typing import Union

Tensor = torch.Tensor


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules.

    Time t goes from 0 (clean data) to 1 (fully noised).
    """

    @abstractmethod
    def rate(self, t: Tensor) -> Tensor:
        """Rate of change of noise: g(t)"""
        pass

    @abstractmethod
    def total(self, t: Tensor) -> Tensor:
        """Total accumulated noise: integral of g(t) from 0 to t"""
        pass

    def __call__(self, t: Tensor) -> Tensor:
        """Alias for total noise."""
        return self.total(t)


class LogLinearNoise(NoiseSchedule):
    """Log-linear noise schedule for absorbing diffusion.

    Designed so that the probability of staying in the original state
    decreases linearly from 1 to eps as t goes from 0 to 1.

    This means: P(stay) = 1 - (1-eps)*t
    And: total_noise = -log(P(stay)) = -log(1 - (1-eps)*t)
    """

    def __init__(self, eps: float = 1e-3):
        """
        Args:
            eps: Small value to prevent log(0) at t=1.
                 At t=1, P(stay) = eps instead of 0.
        """
        self.eps = eps

    def rate(self, t: Tensor) -> Tensor:
        """Rate of noise injection: g(t) = (1-eps) / (1 - (1-eps)*t)"""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total(self, t: Tensor) -> Tensor:
        """Total noise: -log(1 - (1-eps)*t)"""
        return -torch.log(1 - (1 - self.eps) * t)


class GeometricNoise(NoiseSchedule):
    """Geometric noise schedule.

    Total noise follows: sigma(t) = sigma_min^(1-t) * sigma_max^t

    This gives exponential interpolation between sigma_min and sigma_max.
    """

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1.0):
        """
        Args:
            sigma_min: Minimum noise level (at t=0)
            sigma_max: Maximum noise level (at t=1)
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_ratio = torch.log(torch.tensor(sigma_max / sigma_min))

    def rate(self, t: Tensor) -> Tensor:
        """Rate of change of sigma: d(sigma)/dt"""
        return self.total(t) * self.log_ratio

    def total(self, t: Tensor) -> Tensor:
        """Total noise: sigma_min^(1-t) * sigma_max^t"""
        return self.sigma_min ** (1 - t) * self.sigma_max ** t


def get_noise_schedule(name: str, **kwargs) -> NoiseSchedule:
    """Factory function for noise schedules.

    Args:
        name: "loglinear" or "geometric"
        **kwargs: Additional arguments for the schedule

    Returns:
        NoiseSchedule instance
    """
    schedules = {
        "loglinear": LogLinearNoise,
        "geometric": GeometricNoise,
    }

    if name not in schedules:
        raise ValueError(f"Unknown noise schedule: {name}. Choose from {list(schedules.keys())}")

    return schedules[name](**kwargs)
