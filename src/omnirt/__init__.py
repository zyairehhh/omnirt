"""OmniRT public package interface."""

from omnirt.api import generate
from omnirt.core.types import GenerateRequest, GenerateResult

__all__ = ["GenerateRequest", "GenerateResult", "generate"]

