"""obstacles"""
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


# =========================================================================================================
# Extra objects to add to the scene
def get_vase(
    index,
    layout,
    rot,
    density=0.001,
    size=0.1,
    sink=4e-5,
):
    name = f'vase{index}'
    object = {
        'name': f'vase{index}',
        'size': np.ones(3) * size,
        'type': 'box',
        'density': density,
        'pos': np.r_[layout[name], size - sink],
        'rot': rot,
        'group': GROUP['vase'],
        'rgba': COLOR['vase'],
    }
    return object
