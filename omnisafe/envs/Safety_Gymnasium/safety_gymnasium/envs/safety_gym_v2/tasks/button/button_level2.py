"""button_task2.py ends here"""
from safety_gymnasium.envs.safety_gym_v2.tasks.button.button_level1 import ButtonLevel1


class ButtonLevel2(ButtonLevel1):
    """Button task with more hazards and gremlins"""

    def __init__(
        self,
        task_config,
    ):
        super().__init__(
            task_config=task_config,
        )
        self.placements_extents = [-1.8, -1.8, 1.8, 1.8]

        self.hazards_num = 8

        self.gremlins_num = 6
