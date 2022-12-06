from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


@dataclass
class Walls:
    # Walls - barriers in the environment not associated with any constraint
    # NOTE: this is probably best to be auto-generated than manually specified
    num: int = 0  # Number of walls
    placements: list = None  # This should not be used
    locations: list = field(default_factory=list)  # This should be used and length == walls_num
    keepout: float = 0.0  # This should not be used
