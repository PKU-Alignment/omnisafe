import omnisafe
import torch
import os,sys

from omnisafe.common.experiment_grid import ExperimentGrid


def train(exp_id, algo, env_id, custom_cfgs):
    torch.set_num_threads(6)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    USE_REDIRECTION = True
    if USE_REDIRECTION:
        if not os.path.exists(custom_cfgs['data_dir']):
            os.makedirs(custom_cfgs['data_dir'])
        sys.stdout = open(f'{custom_cfgs["data_dir"]}terminal.log', 'w')
        sys.stderr = open(f'{custom_cfgs["data_dir"]}error.log', 'w')
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


if __name__ == "__main__":
    eg = ExperimentGrid(exp_name="Safety_Gymnasium_Goal")
    eg.add('algo', ['PPO', 'PPOLag'])
    eg.add('env_id', ['SafetyPointGoal1-v0'])
    eg.add('epochs', 100)
    eg.add('actor_lr', [0.001, 0.003, 0.004], 'lr', True)
    eg.add('actor_iters', [1, 2], 'ac_iters', True)
    eg.add('seed', [0, 5, 10])
    # eg.add('env_cfgs:num_envs', [2, 3])
    eg.run(train, num_pool=5)
