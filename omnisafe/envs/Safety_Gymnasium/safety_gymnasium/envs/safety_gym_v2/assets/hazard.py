"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


def get_hazard(index, layout, rot, size=0.3):
    name = f'hazard{index}'
    geom = {
        'name': name,
        'size': [size, 1e-2],  # self.hazards_size / 2],
        'pos': np.r_[layout[name], 2e-2],  # self.hazards_size / 2 + 1e-2],
        'rot': rot,
        'type': 'cylinder',
        'contype': 0,
        'conaffinity': 0,
        'group': GROUP['hazard'],
        'rgba': COLOR['hazard'] * [1, 1, 1, 0.25],
    }  # 0.1]}  # transparent
    return geom
