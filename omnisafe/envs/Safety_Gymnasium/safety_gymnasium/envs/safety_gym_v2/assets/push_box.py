"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_push_box(layout, rot, density=0.001, size=0.2):
    object = {
        'name': 'box',
        'type': 'box',
        'size': np.ones(3) * size,
        'pos': np.r_[layout['box'], size],
        'rot': rot,
        'density': density,
        'group': GROUP['box'],
        'rgba': COLOR['box'],
    }
    return object
