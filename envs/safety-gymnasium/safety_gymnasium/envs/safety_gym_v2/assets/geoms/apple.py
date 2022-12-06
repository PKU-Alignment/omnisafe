from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


@dataclass
class Apples:
    # Apples and Oranges are as same as Goal.
    # While they can be instantiated more than one.
    # And one can define different rewards for Apple and Orange.
    num: int = 0
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.3
