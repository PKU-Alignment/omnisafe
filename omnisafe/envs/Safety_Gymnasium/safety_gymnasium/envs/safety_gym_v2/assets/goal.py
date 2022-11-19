"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


# =========================================================================================================
# Extra geoms (immovable objects) to add to the scene
def get_goal(layout, rot, size=0.3):
    geom = {
        'name': 'goal',
        'size': [size, size / 2],
        'pos': np.r_[layout['goal'], size / 2 + 1e-2],
        'rot': rot,
        'type': 'cylinder',
        'contype': 0,
        'conaffinity': 0,
        'group': GROUP['goal'],
        'rgba': COLOR['goal'] * [1, 1, 1, 0.25],
    }  # transparent
    return geom
