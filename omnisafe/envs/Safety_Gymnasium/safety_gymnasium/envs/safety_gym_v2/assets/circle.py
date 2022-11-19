"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


# =====================================================================
# CircleTask specific


def get_circle(radius=1.5):
    geom = {
        'name': 'circle',
        'size': np.array([radius, 1e-2]),
        'pos': np.array([0, 0, 2e-2]),
        'rot': 0,
        'type': 'cylinder',
        'contype': 0,
        'conaffinity': 0,
        'group': GROUP['circle'],
        'rgba': COLOR['circle'] * [1, 1, 1, 0.1],
    }
    return geom
