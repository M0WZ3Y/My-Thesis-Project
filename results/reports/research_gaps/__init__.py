"""
Research Gaps Analysis Package
Addressing critical gaps in PJM electricity price prediction literature
"""

from .unified_model_comparison import UnifiedModelComparison
from .model_explainability import ModelExplainability
from .feature_fusion import FeatureFusion

__all__ = [
    'UnifiedModelComparison',
    'ModelExplainability', 
    'FeatureFusion'
]

__version__ = "1.0.0"
__author__ = "PJM Research Team"
__description__ = "Research gap analysis for electricity price prediction"