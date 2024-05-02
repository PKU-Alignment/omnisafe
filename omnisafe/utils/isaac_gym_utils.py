# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Utils for making Safe Isaac Gym environments."""


from __future__ import annotations

import argparse
from typing import Any

import torch
from isaacgym import gymapi, gymutil
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.vec_task import VecTaskPython
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import (
    ShadowHandCatchOver2Underarm_Safe_finger as ShadowHandCatchOver2UnderarmSafeFinger,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import (
    ShadowHandCatchOver2Underarm_Safe_joint as ShadowHandCatchOver2UnderarmSafeJoint,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import (
    ShadowHandOver_Safe_finger as ShadowHandOverSafeFinger,
)
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import (
    ShadowHandOver_Safe_joint as ShadowHandOverSafeJoint,
)

from omnisafe.typing import DEVICE_CPU


class GymnasiumIsaacEnv(VecTaskPython):
    """This wrapper will use Gymnasium API to wrap Isaac Gym environment."""

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment."""
        obs, rews, costs, terminated, infos = super().step(action.unsqueeze(0))
        truncated = terminated
        return (
            obs.squeeze(0),
            rews.squeeze(0),
            costs.squeeze(0),
            terminated.squeeze(0),
            truncated.squeeze(0),
            infos,
        )

    def reset(self) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment."""
        obs = super().reset()
        return obs.squeeze(0), {}


def parse_sim_params(args: argparse.Namespace) -> gymapi.SimParams:
    """Set up parameters for simulation."""
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != 'cpu':
            print('WARNING: Using Flex with GPU instead of PHYSX!')
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline if args.device != 'cpu' else False
    sim_params.physx.use_gpu = args.use_gpu if args.device != 'cpu' else False

    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def make_isaac_gym_env(
    env_id: str,
    num_envs: int,
    device: torch.device = DEVICE_CPU,
) -> GymnasiumIsaacEnv:
    """Creates and initializes an Isaac Gym environment with specified configurations.

    Args:
        env_id (str): Identifier for the specific environment to be instantiated.
        num_envs (int): The number of parallel environments to create.
        device (torch.device, optional): The computational device ('cpu' or 'cuda:device_id').

    Returns:
        GymnasiumIsaacEnv: An initialized Isaac Gym environment object.
    """
    custom_parameters = [
        {'name': '--algo', 'type': str, 'default': 'PPOLag'},
        {'name': '--env-id', 'type': str, 'default': 'ShadowHandOver_Safe_finger'},
        {'name': '--parallel', 'type': int, 'default': 1},
        {'name': '--seed', 'type': int, 'default': 0},
        {'name': '--total-steps', 'type': int, 'default': 100000000},
        {'name': '--device', 'type': str, 'default': 'cpu'},
        {'name': '--vector-env-nums', 'type': int, 'default': 256},
        {'name': '--torch-threads', 'type': int, 'default': 16},
    ]
    args = gymutil.parse_arguments(custom_parameters=custom_parameters)
    args.device = args.sim_device_type if args.use_gpu_pipeline and args.device != 'cpu' else 'cpu'
    sim_params = parse_sim_params(args=args)

    device_id = int(str(device).rsplit(':', maxsplit=1)[-1]) if str(device) != 'cpu' else 0

    rl_device = device
    if env_id == 'ShadowHandCatchOver2UnderarmSafeFinger':
        task_fn = ShadowHandCatchOver2UnderarmSafeFinger
    elif env_id == 'ShadowHandCatchOver2UnderarmSafeJoint':
        task_fn = ShadowHandCatchOver2UnderarmSafeJoint
    elif env_id == 'ShadowHandOverSafeFinger':
        task_fn = ShadowHandOverSafeFinger
    elif env_id == 'ShadowHandOverSafeJoint':
        task_fn = ShadowHandOverSafeJoint
    else:
        raise NotImplementedError

    task = task_fn(
        num_envs=num_envs,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=True,
        is_multi_agent=False,
    )

    return GymnasiumIsaacEnv(task, rl_device)
