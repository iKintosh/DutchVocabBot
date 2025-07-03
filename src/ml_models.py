"""
Machine Learning Models - Backward Compatibility Module.

This module provides imports for the refactored ML components.
"""

# Import the refactored classes for backward compatibility
from ml.contextual_bandits import ContextualBandits
from ml.features import calculate_word_difficulty
from ml.progress_predictor import LearningProgressPredictor

# Export the main classes
__all__ = ["LearningProgressPredictor", "ContextualBandits", "calculate_word_difficulty"]
