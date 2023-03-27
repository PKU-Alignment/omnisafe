# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the command interfaces."""

import os
import sys
import warnings
from typing import List

import numpy as np
import typer
import yaml
from rich.console import Console

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import NamedTuple, Tuple
from omnisafe.utils.tools import assert_with_exit, custom_cfgs_to_dict, update_dic


app = typer.Typer()
console = Console()


@app.command()
def train(  # pylint: disable=too-many-arguments
    algo: str = 'PPOLag',
    env_id: str = 'SafetyHumanoidVelocity-v4',
    parallel: int = 1,
    total_steps: int = 1638400,
    device: str = 'cpu',
    vector_env_nums: int = 16,
    torch_threads: int = 16,
    log_dir: str = os.path.join(os.getcwd()),
    custom_cfgs: List[str] = [],
):
    """Train a single policy in OmniSafe via command line.

    Example:

    .. code-block:: bash

        python -m omnisafe train --algo PPOLag --env_id SafetyPointGoal1-v0 --parallel 1
        --total_steps 1000000 --device cpu --vector_env_nums 1

    Args:
        algo: algorithm to train
        env_id: the name of test environment
        parallel: number of paralleled progress for calculations.
        total_steps: total number of steps to train for algorithm
        device: device to use for training
        vector_env_nums: number of vector envs to use for training
        torch_threads: number of threads to use for torch
        log_dir: directory to save logs, default is current directory
        custom_cfgs: custom configuration for training
    """
    args = {
        'algo': algo,
        'env_id': env_id,
        'parallel': parallel,
        'total_steps': total_steps,
        'device': device,
        'vector_env_nums': vector_env_nums,
        'torch_threads': torch_threads,
    }
    keys = custom_cfgs[0::2]
    values = list(custom_cfgs[1::2])
    custom_cfgs = dict(zip(keys, values))
    custom_cfgs.update({'logger_cfgs:log_dir': os.path.join(log_dir, 'train')})

    parsed_custom_cfgs = {}
    for k, v in custom_cfgs.items():
        update_dic(parsed_custom_cfgs, custom_cfgs_to_dict(k, v))

    agent = omnisafe.Agent(
        algo=algo,
        env_id=env_id,
        train_terminal_cfgs=args,
        custom_cfgs=parsed_custom_cfgs,
    )
    agent.learn()


def train_grid(
    exp_id: str, algo: str, env_id: str, custom_cfgs: NamedTuple
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Example:

    .. code-block:: bash

        python -m omnisafe train_grid --exp_id exp-1 --algo PPOLag --env_id SafetyPointGoal1-v0
        --parallel 1 --total_steps 1000000 --device cpu --vector_env_nums 1

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (NamedTuple): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'])
    # pylint: disable-next=consider-using-with
    sys.stdout = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', terminal_log_name),
        'w',
        encoding='utf-8',
    )
    # pylint: disable-next=consider-using-with
    sys.stderr = open(
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', error_log_name),
        'w',
        encoding='utf-8',
    )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


@app.command()
def benchmark(
    exp_name: str = typer.Argument(..., help='experiment name'),
    num_pool: int = typer.Argument(..., help='number of paralleled experiments.'),
    config_path: str = typer.Argument(
        ..., help='path to config file, it is supposed to be yaml file, e.g. ./configs/ppo.yaml'
    ),
    log_dir: str = typer.Option(
        os.path.join(os.getcwd()), help='directory to save logs, default is current directory'
    ),
):
    """Benchmark algorithms configured by .yaml file in OmniSafe via command line.

    Example:

    .. code-block:: bash

        python -m omnisafe benchmark --exp_name exp-1 --num_pool 1 --config_path ./configs/
        on-policy/PPOLag.yaml--log_dir ./runs

    Args:
        exp_name: experiment name
        num_pool: number of paralleled experiments.
        config_path: path to config file, it is supposed to be yaml file
        log_dir: directory to save logs, default is current directory
    """
    assert_with_exit(config_path.endswith('.yaml'), 'config file must be yaml file')
    with open(config_path, encoding='utf-8') as file:
        try:
            configs = yaml.load(file, Loader=yaml.FullLoader)
            assert configs is not None, 'load file error'
        except yaml.YAMLError as exc:
            assert False, f'load file error: {exc}'
    assert_with_exit('algo' in configs, '`algo` must be specified in config file')
    assert_with_exit('env_id' in configs, '`env_id` must be specified in config file')
    if np.prod([len(v) if isinstance(v, list) else 1 for v in configs.values()]) % num_pool != 0:
        warnings.warn(
            'In order to maximize the use of computational resources, '
            'total number of experiments should be evenly divided by `num_pool`'
        )
    log_dir = os.path.join(log_dir, 'benchmark')
    eg = ExperimentGrid(exp_name=exp_name)
    for k, v in configs.items():
        eg.add(key=k, vals=v)
    eg.run(train_grid, num_pool=num_pool, parent_dir=log_dir)


@app.command('eval')
def evaluate(
    result_dir: str = typer.Argument(
        ...,
        help='directory of experiment results to evaluate, e.g. ./runs/PPO-{SafetyPointGoal1-v0}',
    ),
    num_episode: int = typer.Option(10, help='number of episodes to render'),
    render: bool = typer.Option(True, help='whether to render'),
    render_mode: str = typer.Option(
        'rgb_array',
        help="render mode('human', 'rgb_array', 'rgb_array_list', 'depth_array', 'depth_array_list')",
    ),
    camera_name: str = typer.Option('track', help='camera name to render'),
    width: int = typer.Option(256, help='width of rendered image'),
    height: int = typer.Option(256, help='height of rendered image'),
):
    """Evaluate a policy which trained by OmniSafe via command line.

    Example:

    .. code-block:: bash

        python -m omnisafe eval --result_dir ./runs/PPOLag-{SafetyPointGoal1-v0} --num_episode 10
        --render True --render_mode rgb_array --camera_name track --width 256 --height 256

    Args:
        result_dir (str): Directory of experiment results to evaluate.
        num_episode (int, optional): Number of episodes to render. Defaults to 10.
        render (bool, optional): Whether to render. Defaults to True.
        render_mode (str, optional): Render mode('human', 'rgb_array', 'rgb_array_list',
        'depth_array', 'depth_array_list'). Defaults to 'rgb_array'.
        camera_name (str, optional): Camera name to render. Defaults to 'track'.
        width (int, optional): Width of rendered image. Defaults to 256.
        height (int, optional): Height of rendered image. Defaults to 256.
    """
    evaluator = omnisafe.Evaluator(render_mode=render_mode)
    assert_with_exit(os.path.exists(result_dir), f'path{result_dir}, no torch_save directory')
    for seed_dir in os.scandir(result_dir):
        if seed_dir.is_dir():
            models_dir = os.path.join(seed_dir.path, 'torch_save')
            for item in os.scandir(models_dir):
                if item.is_file() and item.name.split('.')[-1] == 'pt':
                    evaluator.load_saved(
                        save_dir=seed_dir,
                        model_name=item.name,
                        camera_name=camera_name,
                        width=width,
                        height=height,
                    )
                    if render:
                        evaluator.render(num_episodes=num_episode)
                    else:
                        evaluator.evaluate(num_episodes=num_episode)


@app.command()
def train_config(
    config_path: str = typer.Argument(
        ..., help='path to config file, it is supposed to be yaml file, e.g. ./configs/ppo.yaml'
    ),
    log_dir: str = typer.Option(
        os.path.join(os.getcwd()), help='directory to save logs, default is current directory'
    ),
):
    """Train a policy configured by .yaml file in OmniSafe via command line.

    Example:

    .. code-block:: bash

        python -m omnisafe train_config --config_path ./configs/on-policy/PPOLag.yaml --log_dir ./
        runs

    Args:
        config_path (str): path to config file, it is supposed to be yaml file.
        log_dir (str, optional): directory to save logs, default is current directory.
        Defaults to os.path.join(os.getcwd()).
    """
    assert_with_exit(config_path.endswith('.yaml'), 'config file must be yaml file')
    with open(config_path, encoding='utf-8') as file:
        try:
            args = yaml.load(file, Loader=yaml.FullLoader)
            assert args is not None, 'load file error'
        except yaml.YAMLError as exc:
            assert False, f'load file error: {exc}'
    assert_with_exit('algo' in args, '`algo` must be specified in config file')
    assert_with_exit('env_id' in args, '`env_id` must be specified in config file')

    args.update({'logger_cfgs': {'log_dir': os.path.join(log_dir, 'train_dict')}})
    agent = omnisafe.Agent(algo=args['algo'], env_id=args['env_id'], custom_cfgs=args)
    agent.learn()


if __name__ == '__main__':
    app()
