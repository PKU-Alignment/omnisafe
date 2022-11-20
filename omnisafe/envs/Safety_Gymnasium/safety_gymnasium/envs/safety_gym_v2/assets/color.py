"""Color"""
import numpy as np


COLOR = {
    # Distinct colors for different types of objects.
    # For now this is mostly used for visualization.
    # This also affects the vision observation, so if training from pixels.
    'box': np.array([1, 1, 0, 1]),
    'button': np.array([1, 0.5, 0, 1]),
    'goal': np.array([0, 1, 0, 1]),
    'vase': np.array([0, 1, 1, 1]),
    'hazard': np.array([0, 0, 1, 1]),
    'pillar': np.array([0.5, 0.5, 1, 1]),
    'wall': np.array([0.5, 0.5, 0.5, 1]),
    'gremlin': np.array([0.5, 0, 1, 1]),
    'circle': np.array([0, 1, 0, 1]),
    'red': np.array([1, 0, 0, 1]),
    'apple': np.array([0.835, 0.169, 0.169, 1]),
    'orange': np.array([1, 0.6, 0, 1]),
}
