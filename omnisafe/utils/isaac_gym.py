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


from __future__ import annotations

import os

import argparse

from isaacgym import gymutil

from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.vec_task import VecTaskPython
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_finger import ShadowHandCatchOver2Underarm_Safe_finger
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandCatchOver2underarm_Safe_joint import ShadowHandCatchOver2Underarm_Safe_joint
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_finger import ShadowHandOver_Safe_finger
from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.ShadowHandOver_Safe_joint import ShadowHandOver_Safe_joint

class GymnasiumIsaacEnv(VecTaskPython):
    """This wrapper will use Gymnasium API to wrap IsaacGym environment."""

    def step(self, action):
        """Steps through the environment."""
        obs, rews, costs, terminated, infos = super().step(action)
        truncated = terminated
        return obs, rews, costs, terminated, truncated, infos
    
    def reset(self):
        """Resets the environment."""
        obs = super().reset()
        return obs, {}
    
def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    try:
        from isaacgym import gymapi, gymutil
    except ImportError:
        raise Exception("Please install isaacgym to run Isaac Gym tasks!")
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
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

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params
    
def make_isaac_gym_env(cfgs, sim_params):
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")

    # Parse arguments

    args = parser.parse_args()
    cfg_env={}
    yaml_path = os.path.abspath(__file__).replace("utils", "multi_agent").replace(".py", ".yaml")
    try:
        from isaacgym import gymutil
    except ImportError:
        raise Exception("Please install isaacgym to run Isaac Gym tasks!")
    args = gymutil.parse_arguments(description="RL Policy")
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device
    task = eval(args.task)(
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        is_multi_agent=False)
    try:
        env = GymnasiumIsaacEnv(task, rl_device)
    except ModuleNotFoundError:
        env = None

    return env