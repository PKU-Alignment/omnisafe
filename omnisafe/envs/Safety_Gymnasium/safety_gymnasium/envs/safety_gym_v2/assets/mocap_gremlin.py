"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


# =========================================================================================================
# Extra mocap bodies used for control (equality to object of same name)
def get_mocap_gremlin(index, layout, rot, size=0.1):
    name = f'gremlin{index}mocap'
    mocap = {
        'name': name,
        'size': np.ones(3) * size,
        'type': 'box',
        'pos': np.r_[layout[name.replace('mocap', '')], size],
        'rot': rot,
        'group': GROUP['gremlin'],
        'rgba': np.array([1, 1, 1, 0.1]) * COLOR['gremlin'],
    }
    return mocap
