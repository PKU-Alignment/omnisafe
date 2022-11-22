"""algo wrapper"""
import os
import sys

import psutil

# from omnisafe.algos import REGISTRY
from omnisafe.algos.registry import REGISTRY
from omnisafe.algos.utils.distributed_tools import mpi_fork
from omnisafe.algos.utils.tools import get_default_kwargs_yaml
from omnisafe.evaluator import Evaluator


# from omnisafe.algos import off_policy, on_policy


class AlgoWrapper:
    """Algo Wrapper for algo"""

    def __init__(self, algo, env, parallel=1, custom_cfgs=None):
        self.algo = algo
        self.env = env
        self.env_id = env.env_id
        self.seed = 0  # TOOD
        self.parallel = parallel
        self.custom_cfgs = custom_cfgs
        self.evaluator = None

    def recursive_update(self, args: dict, update_args: dict):
        """recursively update args"""
        for key, value in args.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    print(f'{key}:')
                    self.recursive_update(args[key], update_args[key])
                else:
                    # f-strings:
                    # https://pylint.pycqa.org/en/latest/user_guide/messages/convention/consider-using-f-string.html
                    args[key] = update_args[key]
                    menus = (key, update_args[key])
                    print(f'- {menus[0]}: {menus[1]} is update!')
            elif isinstance(value, dict):
                self.recursive_update(value, update_args)

    def learn(self):
        """Agent Learning"""
        # Use number of physical cores as default.
        # If also hardware threading CPUs should be used
        # enable this by the use_number_of_threads=True
        physical_cores = psutil.cpu_count(logical=False)
        use_number_of_threads = bool(self.parallel > physical_cores)

        if mpi_fork(self.parallel, use_number_of_threads=use_number_of_threads):
            # Re-launches the current script with workers linked by MPI
            sys.exit()

        agent = None
        on_policy_list = [
            'PolicyGradient',
            'PPO',
            'PPOLag',
            'NaturalPG',
            'TRPO',
            'TRPOLag',
            'PDO',
            'NPGLag',
            'CPO',
            'RCPO',
            'CRPO',
            'PCPO',
            'P3O',
            'IPO',
            'FOCOPS',
            'CPPOPid',
        ]
        off_policy_list = [
            'DDPG',
            'DDPGLag',
            'TD3',
            'TD3Lag',
            'SAC',
            'SACLag',
            'CVPO',
        ]
        assert self.algo in on_policy_list + off_policy_list, f'{self.algo} is not supported!'
        on_policy_flag = self.algo in on_policy_list
        cfgs = get_default_kwargs_yaml(self.algo, self.env_id, on_policy_flag)
        if self.custom_cfgs is not None:
            self.recursive_update(cfgs, self.custom_cfgs)
        exp_name = os.path.join(self.env.env_id, self.algo)
        cfgs.update(exp_name=exp_name)
        agent = REGISTRY.get(self.algo)(
            env=self.env,
            exp_name=exp_name,
            data_dir=cfgs['data_dir'],
            seed=cfgs['seed'],
            cfgs=cfgs,
        )
        ac = agent.learn()

        self.evaluator = Evaluator(self.env, ac.pi, ac.obs_oms)

    def evaluate(self, num_episodes: int = 10, horizon: int = 1000, cost_criteria: float = 1.0):
        """Agent Evaluation"""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.evaluate(num_episodes, horizon, cost_criteria)

    def render(
        self,
        num_episode: int = 0,
        horizon: int = 1000,
        seed: int = None,
        play=True,
        save_replay_path: str = None,
    ):
        """Render the environment."""
        assert self.evaluator is not None, 'Please run learn() first!'
        self.evaluator.render(num_episode, horizon, seed, play, save_replay_path)
