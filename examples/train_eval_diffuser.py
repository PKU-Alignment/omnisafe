import datetime
import gym
import numpy as np

import omnisafe
import gym_examples

import torch
from omnisafe.algorithms.offline.decision_diffuser import DecisionDiffuser 
from omnisafe.models.actor import DecisionDiffuserActor
from omnisafe import envs
import matplotlib.pyplot as plt

DEVICE = "cpu"

env_id ='gym_examples/CondCircle-v0'
agent = omnisafe.Agent('DecisionDiffuser', env_id)
agent.learn()

env = envs.make(env_id)
agent: DecisionDiffuser = agent.agent
actor: DecisionDiffuserActor = agent._actor

def cls_free_cond(actor):
    """Sample from the model with cls free condition
    generate statisfy both
    cond1 is outside the circle of radius 1.2
    cond2 is inside the circle of radius 1.5
    """
    cls_free_cond1 = torch.tensor([[1.0,0.0]], device=DEVICE)
    cls_free_cond2 = torch.tensor([[0.0,1.0]], device=DEVICE)
    for i in range(1):
        state_cond = torch.tensor([[np.random.uniform(1, 4), np.random.uniform(1, 4)]], device=DEVICE)
        state_cond = {0: state_cond}
        x = actor._model.conditional_sample(state_cond, cls_free_condition_list=[cls_free_cond1, cls_free_cond2])
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(xys[:, 0], xys[:, 1],  c=np.arange(len(xys[:,0])), cmap='viridis',)
    plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")

def state_cond(actor):
    """Sample from the model with state condition, generate a plan from random starting point to (4, 1)"""
    for i in range(1):
        state_cond = torch.tensor([[np.random.uniform(1, 4), np.random.uniform(1, 4)]], device=DEVICE)
        state_cond = {0: state_cond, 79: torch.tensor([[4.0,1.0]], device=DEVICE)}
        dummy_cls_cond = torch.tensor([[0.0,0.0]], device=DEVICE)
        x = actor._model.conditional_sample(state_cond, cls_free_condition=[dummy_cls_cond])
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(xys[:, 0], xys[:, 1],  c=np.arange(len(xys[:,0])), cmap='viridis',)
    plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")

def both_cond(actor):
    """
    condition on both state and cls free condition
    
    
    """
    for i in range(1):
        state_cond = torch.tensor([[np.random.uniform(1, 4), np.random.uniform(1, 4)]], device=DEVICE)
        state_cond = {0: state_cond, 79: torch.tensor([[4.0,1.0]], device=DEVICE)}
        cls_cond = torch.tensor([[1.0,0.0]], device=DEVICE)
        x = actor._model.conditional_sample(state_cond, cls_free_condition=[cls_cond])
        obs = x.cpu().numpy()
        xys = obs[0, :, 0:2]
        plt.scatter(xys[:, 0], xys[:, 1],  c=np.arange(len(xys[:,0])), cmap='viridis',)
    plt.savefig(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".png")