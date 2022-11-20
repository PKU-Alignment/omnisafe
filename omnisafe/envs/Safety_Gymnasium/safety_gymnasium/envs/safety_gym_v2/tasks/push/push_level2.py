"""push_level2"""
import mujoco
from safety_gymnasium.envs.safety_gym_v2.tasks.push.push_level1 import PushLevel1


class PushLevel2(PushLevel1):
    """
    A task where agents have to run as fast as possible within a circular zone.
    Rewards are by default shaped.
    """

    def __init__(
        self,
        task_config,
    ):
        super().__init__(
            task_config=task_config,
        )
        self.placements_extents = [-2, -2, 2, 2]

        self.hazards_num = 4

        self.pillars_num = 4

    def calculate_cost(self, **kwargs):
        """determine costs depending on agent and obstacles"""
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Conctacts processing
        cost['cost_pillars'] = 0

        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if any(n.startswith('pillar') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_pillars'] += self.pillars_cost

        # Calculate constraint violations
        cost['cost_hazards'] = 0
        for h_pos in self.hazards_pos:
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.hazards_size:
                cost['cost_hazards'] += self.hazards_cost * (self.hazards_size - h_dist)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost
