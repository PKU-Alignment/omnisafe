"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_gremlin(
    index,
    layout,
    rot,
    density=0.001,
    size=0.1,
):
    name = f'gremlin{index}obj'
    object = {
        'name': name,
        'size': np.ones(3) * size,
        'type': 'box',
        'density': density,
        'pos': np.r_[layout[name.replace('obj', '')], size],
        'rot': rot,
        'group': GROUP['gremlin'],
        'rgba': COLOR['gremlin'],
    }
    return object
