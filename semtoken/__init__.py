"""
SemToken: Semantic-Aware Tokenization for Efficient Long-Context Language Modeling

A semantic-aware tokenization framework that dynamically adjusts token granularity 
based on local semantic density and reduces token redundancy through intelligent clustering.
"""

from .core import SemToken
from .semantic_encoder import SemanticEncoder
from .clustering import SemanticCluster
from .granularity import GranularityAssigner
from .budget import BudgetAllocator
from .utils import TokenMerger, SemanticAnalyzer

__version__ = "1.0.0"
__author__ = "Dong Liu, Yanxuan Yu"

__all__ = [
    "SemToken",
    "SemanticEncoder", 
    "SemanticCluster",
    "GranularityAssigner",
    "BudgetAllocator",
    "TokenMerger",
    "SemanticAnalyzer"
]
