"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_button(index, layout, rot, size=0.1):
    name = f'button{index}'
    geom = {
        'name': name,
        'size': np.ones(3) * size,
        'pos': np.r_[layout[name], size],
        'rot': rot,
        'type': 'sphere',
        'group': GROUP['button'],
        'rgba': COLOR['button'],
    }
    return geom
