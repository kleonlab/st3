"""
Score-Entropy Discrete Diffusion (SEDD) for RNA-seq Data

A clean implementation of discrete diffusion for masked gene expression prediction
on single-cell RNA-seq data.
"""

from .noise import LogLinearNoise, GeometricNoise, get_noise_schedule
from .graph import AbsorbingGraph, UniformGraph, get_graph
from .model import SEDDTransformer, SEDDTransformerSmall, SEDDTransformerMedium, SEDDTransformerLarge
from .sampling import EulerSampler, AnalyticSampler, impute_masked, get_sampler
from .trainer import SEDDTrainer, create_trainer
from .data import (
    RNASeqDataset,
    discretize_expression,
    undiscretize_expression,
    create_synthetic_rnaseq,
    train_val_split,
    normalize_expression,
)

__version__ = "0.1.0"
__all__ = [
    # Noise
    "LogLinearNoise",
    "GeometricNoise",
    "get_noise_schedule",
    # Graph
    "AbsorbingGraph",
    "UniformGraph",
    "get_graph",
    # Model
    "SEDDTransformer",
    "SEDDTransformerSmall",
    "SEDDTransformerMedium",
    "SEDDTransformerLarge",
    # Sampling
    "EulerSampler",
    "AnalyticSampler",
    "impute_masked",
    "get_sampler",
    # Training
    "SEDDTrainer",
    "create_trainer",
    # Data
    "RNASeqDataset",
    "discretize_expression",
    "undiscretize_expression",
    "create_synthetic_rnaseq",
    "train_val_split",
    "normalize_expression",
]
