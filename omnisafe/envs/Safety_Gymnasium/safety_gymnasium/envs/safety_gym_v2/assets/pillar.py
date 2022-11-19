"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_pillar(index, layout, rot, size=0.2, height=0.5):
    name = f'pillar{index}'
    geom = {
        'name': name,
        'size': [size, height],
        'pos': np.r_[layout[name], height],
        'rot': rot,
        'type': 'cylinder',
        'group': GROUP['pillar'],
        'rgba': COLOR['pillar'],
    }
    return geom
