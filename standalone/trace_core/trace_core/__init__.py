"""Baseline-agnostic TRACE orchestration core."""

from .config import BaselineAdapter, PipelineConfig, load_configuration

__all__ = ["BaselineAdapter", "PipelineConfig", "load_configuration"]
