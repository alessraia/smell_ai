"""LLM-based smell detection (CR01).

This package is additive and kept isolated from the standard AST-based pipeline.
Integration points (GUI / analyzer / report) will opt-in to these components.
"""

from .types import (
    LLMCatalog,
    LLMSmellDefinition,
    LLMProviderDefinition,
    LLMSmellFinding,
    DetectionTarget,
    PromptMode,
    ProviderKind,
)
from .providers import LLMProvider, MockLLMProvider, LocalLLMProvider, ApiLLMProvider
from .catalog_store import LLMCatalogStore
from .catalog_service import LLMCatalogService, CatalogValidationError
from .orchestrator import LLMOrchestrator

__all__ = [
    "LLMCatalog",
    "LLMSmellDefinition",
    "LLMProviderDefinition",
    "LLMSmellFinding",
    "DetectionTarget",
    "PromptMode",
    "ProviderKind",
    "LLMProvider",
    "MockLLMProvider",
    "LocalLLMProvider",
    "ApiLLMProvider",
    "LLMCatalogStore",
    "LLMCatalogService",
    "CatalogValidationError",
    "LLMOrchestrator",
]
