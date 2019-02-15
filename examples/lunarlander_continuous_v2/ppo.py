# -*- coding: utf-8 -*-
"""Run module for PPO on LunarLanderContinuous-v2.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import argparse

import gym
import numpy as np
import torch
import torch.optim as optim

from algorithms.common.networks.mlp import MLP, GaussianDist
from algorithms.ppo.agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
hyper_params = {
    "GAMMA": 0.95,
    "LAMBDA": 0.9,
    "EPSILON": 0.2,
    "W_VALUE": 0.5,
    "W_ENTROPY": 1e-3,
    "LR_ACTOR": 3e-4,
    "LR_CRITIC": 3e-4,
    "LR_ENTROPY": 1e-6,
    "EPOCH": 4,
    "BATCH_SIZE": 3,
    "ROLLOUT_LEN": 12,
    "WEIGHT_DECAY": 0.0,
    "AUTO_ENTROPY_TUNING": True,
}


def run(env: gym.Env, args: argparse.Namespace, state_dim: int, action_dim: int):
    """Run training or test.

    Args:
        env (gym.Env): openAI Gym environment with continuous action space
        args (argparse.Namespace): arguments including training settings
        state_dim (int): dimension of states
        action_dim (int): dimension of actions

    """
    hidden_sizes_actor = [256]
    hidden_sizes_critic = [256]

    # target entropy
    target_entropy = -np.prod((action_dim,)).item()  # heuristic

    # create models
    actor = GaussianDist(
        input_size=state_dim, output_size=action_dim, hidden_sizes=hidden_sizes_actor
    ).to(device)

    critic = MLP(
        input_size=state_dim, output_size=1, hidden_sizes=hidden_sizes_critic
    ).to(device)

    # create optimizer
    actor_optim = optim.Adam(
        actor.parameters(),
        lr=hyper_params["LR_ACTOR"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    critic_optim = optim.Adam(
        critic.parameters(),
        lr=hyper_params["LR_CRITIC"],
        weight_decay=hyper_params["WEIGHT_DECAY"],
    )

    # make tuples to create an agent
    models = (actor, critic)
    optims = (actor_optim, critic_optim)

    # create an agent
    agent = Agent(env, args, hyper_params, models, optims, target_entropy)

    # run
    if args.test:
        agent.test()
    else:
        agent.train()
