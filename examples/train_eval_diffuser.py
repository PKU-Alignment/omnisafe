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
"""Example of using DecisionDiffuser to generate plans with different conditions."""

import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

import omnisafe
from omnisafe import envs
from omnisafe.algorithms.offline.decision_diffuser import DecisionDiffuser
from omnisafe.models.actor import DecisionDiffuserActor


DEVICE = 'cpu'

env_id = 'gym_examples/CondCircle-v0'
agent = omnisafe.Agent('DecisionDiffuser', env_id)
agent.learn()

env = envs.make(env_id)
agent: DecisionDiffuser = agent.agent
actor: DecisionDiffuserActor = agent._actor


def cls_free_cond(actor: DecisionDiffuserActor) -> None:
    """
    Sample from the model with cls free condition.
    Generate samples that satisfy both conditions:
    - cond1: points outside the circle of radius 1.2
    - cond2: points inside the circle of radius 1.5

    Args:
        actor (DecisionDiffuserActor): The actor object used for conditional sampling.
    """
    cls_free_cond1 = torch.tensor([[1.0, 0.0]], device=DEVICE)
    cls_free_cond2 = torch.tensor([[0.0, 1.0]], device=DEVICE)
    for _ in range(1):
        state_cond = torch.tensor(
            [[np.random.uniform(1, 4), np.random.uniform(1, 4)]],
            device=DEVICE,
        )
        state_cond = {0: state_cond}
        x = actor.model.conditional_sample(
            state_cond,
            cls_free_condition_list=[cls_free_cond1, cls_free_cond2],
        )
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(
            xys[:, 0],
            xys[:, 1],
            c=np.arange(len(xys[:, 0])),
            cmap='viridis',
        )
    plt.savefig(datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png')


def state_cond(actor: DecisionDiffuserActor) -> None:
    """
    Sample from the model with state condition.
    Generate a plan from a random starting point to (4, 1).
    
    Args:
        actor (DecisionDiffuserActor): The actor object used for conditional sampling.
    """
    # Your code here   for _ in range(1):
        state_cond = torch.tensor(
            [[np.random.uniform(1, 4), np.random.uniform(1, 4)]],
            device=DEVICE,
        )
        state_cond = {0: state_cond, 79: torch.tensor([[4.0, 1.0]], device=DEVICE)}
        dummy_cls_cond = torch.tensor([[0.0, 0.0]], device=DEVICE)
        x = actor.model.conditional_sample(state_cond, cls_free_condition=[dummy_cls_cond])
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(
            xys[:, 0],
            xys[:, 1],
            c=np.arange(len(xys[:, 0])),
            cmap='viridis',
        )
    plt.savefig(datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png')


def both_cond(actor: DecisionDiffuserActor) -> None:
    """
    Condition on both state and cls free condition.

    Args:
        actor (DecisionDiffuserActor): The actor object used for conditional sampling.
    """
    for _ in range(1):
        state_cond = torch.tensor(
            [[np.random.uniform(1, 4), np.random.uniform(1, 4)]],
            device=DEVICE,
        )
        state_cond = {0: state_cond, 79: torch.tensor([[4.0, 1.0]], device=DEVICE)}
        cls_cond = torch.tensor([[1.0, 0.0]], device=DEVICE)
        x = actor.model.conditional_sample(state_cond, cls_free_condition=[cls_cond])
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(
            xys[:, 0],
            xys[:, 1],
            c=np.arange(len(xys[:, 0])),
            cmap='viridis',
        )
    plt.savefig(datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.png')
