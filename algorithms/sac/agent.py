# -*- coding: utf-8 -*-
"""SAC agent for episodic tasks in OpenAI Gym.

- Author: Curt Park
- Contact: curt.park@medipixel.io
- Paper: https://arxiv.org/pdf/1801.01290.pdf
         https://arxiv.org/pdf/1812.05905.pdf
"""

import argparse
import os
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

import algorithms.common.helper_functions as common_utils
from algorithms.common.abstract.agent import AbstractAgent
from algorithms.common.buffer.replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent(AbstractAgent):
    """SAC agent interacting with environment.

    Attrtibutes:
        memory (ReplayBuffer): replay memory
        actor (nn.Module): actor model to select actions
        actor_target (nn.Module): target actor model to select actions
        actor_optimizer (Optimizer): optimizer for training actor
        critic_1 (nn.Module): critic model to predict state values
        critic_2 (nn.Module): critic model to predict state values
        critic_target1 (nn.Module): target critic model to predict state values
        critic_target2 (nn.Module): target critic model to predict state values
        critic_optimizer1 (Optimizer): optimizer for training critic_1
        critic_optimizer2 (Optimizer): optimizer for training critic_2
        curr_state (np.ndarray): temporary storage of the current state
        target_entropy (int): desired entropy used for the inequality constraint
        alpha (torch.Tensor): weight for entropy
        alpha_optimizer (Optimizer): optimizer for alpha
        hyper_params (dict): hyper-parameters
        total_step (int): total step numbers
        episode_step (int): step number of the current episode

    """

    def __init__(
        self,
        env: gym.Env,
        args: argparse.Namespace,
        hyper_params: dict,
        models: tuple,
        optims: tuple,
        target_entropy: float,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            args (argparse.Namespace): arguments including hyperparameters and training settings
            hyper_params (dict): hyper-parameters
            models (tuple): models including actor and critic
            optims (tuple): optimizers for actor and critic
            target_entropy (float): target entropy for the inequality constraint

        """
        AbstractAgent.__init__(self, env, args)

        self.actor, self.vf, self.vf_target, self.qf_1, self.qf_2 = models
        self.actor_optimizer, self.vf_optimizer = optims[0:2]
        self.qf_1_optimizer, self.qf_2_optimizer = optims[2:4]
        self.hyper_params = hyper_params
        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0

        # automatic entropy tuning
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.hyper_params["LR_ENTROPY"]
            )

        # load the optimizer and model parameters
        if args.load_from is not None and os.path.exists(args.load_from):
            self.load_params(args.load_from)

        if not self.args.test:
            # replay memory
            self.memory = ReplayBuffer(
                hyper_params["BUFFER_SIZE"], hyper_params["BATCH_SIZE"]
            )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # if initial random action should be conducted
        if (
            self.total_step < self.hyper_params["INITIAL_RANDOM_ACTION"]
            and not self.args.test
        ):
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).to(device)
        if self.args.test and not self.is_discrete:
            _, _, _, selected_action, _ = self.actor(state)
        else:
            selected_action, _, _, _, _ = self.actor(state)

        return selected_action.detach().cpu().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        self.total_step += 1
        self.episode_step += 1

        next_state, reward, done, _ = self.env.step(action)

        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )
            self.memory.add(self.curr_state, action, reward, next_state, done_bool)

        return next_state, reward, done

    def update_model(
        self,
        experiences: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the model after each episode."""
        states, actions, rewards, next_states, dones = experiences
        new_actions, log_prob, pre_tanh_value, mu, std = self.actor(states)

        # train alpha
        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            alpha_loss = (
                -self.log_alpha * (log_prob + self.target_entropy).detach()
            ).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.zeros(1)
            alpha = self.hyper_params["W_ENTROPY"]

        # Q function loss
        masks = 1 - dones
        q_1_pred = self.qf_1(states, actions)
        q_2_pred = self.qf_2(states, actions)
        v_target = self.vf_target(next_states)
        q_target = rewards + self.hyper_params["GAMMA"] * v_target * masks
        qf_1_loss = F.mse_loss(q_1_pred, q_target.detach())
        qf_2_loss = F.mse_loss(q_2_pred, q_target.detach())

        # V function loss
        v_pred = self.vf(states)
        q_pred = torch.min(
            self.qf_1(states, new_actions), self.qf_2(states, new_actions)
        )
        v_target = q_pred - alpha * log_prob
        vf_loss = F.mse_loss(v_pred, v_target.detach())

        # train Q functions
        self.qf_1_optimizer.zero_grad()
        qf_1_loss.backward()
        self.qf_1_optimizer.step()

        self.qf_2_optimizer.zero_grad()
        qf_2_loss.backward()
        self.qf_2_optimizer.step()

        # train V function
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        if self.total_step % self.hyper_params["DELAYED_UPDATE"] == 0:
            # actor loss
            advantage = q_pred - v_pred.detach()
            actor_loss = (alpha * log_prob - advantage).mean()

            # regularization
            if self.c == -1:  # iff the action is continuous
                mean_reg = self.hyper_params["W_MEAN_REG"] * mu.pow(2).mean()
                std_reg = self.hyper_params["W_STD_REG"] * std.pow(2).mean()
                pre_activation_reg = self.hyper_params["W_PRE_ACTIVATION_REG"] * (
                    pre_tanh_value.pow(2).sum(dim=-1).mean()
                )
                actor_reg = mean_reg + std_reg + pre_activation_reg

                # actor loss + regularization
                actor_loss += actor_reg

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update target networks
            common_utils.soft_update(self.vf, self.vf_target, self.hyper_params["TAU"])
        else:
            actor_loss = torch.zeros(1)

        return (
            actor_loss.data,
            qf_1_loss.data,
            qf_2_loss.data,
            vf_loss.data,
            alpha_loss.data,
        )

    def load_params(self, path: str):
        """Load model and optimizer parameters."""
        if not os.path.exists(path):
            print("[ERROR] the input path does not exist. ->", path)
            return

        params = torch.load(path)
        self.actor.load_state_dict(params["actor"])
        self.qf_1.load_state_dict(params["qf_1"])
        self.qf_2.load_state_dict(params["qf_2"])
        self.vf.load_state_dict(params["vf"])
        self.vf_target.load_state_dict(params["vf_target"])
        self.actor_optimizer.load_state_dict(params["actor_optim"])
        self.qf_1_optimizer.load_state_dict(params["qf_1_optim"])
        self.qf_2_optimizer.load_state_dict(params["qf_2_optim"])
        self.vf_optimizer.load_state_dict(params["vf_optim"])

        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            self.alpha_optimizer.load_state_dict(params["alpha_optim"])

        print("[INFO] loaded the model and optimizer from", path)

    def save_params(self, n_episode: int):
        """Save model and optimizer parameters."""
        params = {
            "actor": self.actor.state_dict(),
            "qf_1": self.qf_1.state_dict(),
            "qf_2": self.qf_2.state_dict(),
            "vf": self.vf.state_dict(),
            "vf_target": self.vf_target.state_dict(),
            "actor_optim": self.actor_optimizer.state_dict(),
            "qf_1_optim": self.qf_1_optimizer.state_dict(),
            "qf_2_optim": self.qf_2_optimizer.state_dict(),
            "vf_optim": self.vf_optimizer.state_dict(),
        }

        if self.hyper_params["AUTO_ENTROPY_TUNING"]:
            params["alpha_optim"] = self.alpha_optimizer.state_dict()

        AbstractAgent.save_params(self, params, n_episode)

    def write_log(
        self, i: int, loss: np.ndarray, score: float = 0.0, delayed_update: int = 1
    ):
        """Write log about loss and score"""
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode_step %d, total step %d, total score: %d\n"
            "total loss: %.3f actor_loss: %.3f qf_1_loss: %.3f qf_2_loss: %.3f "
            "vf_loss: %.3f alpha_loss: %.3f\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0] * delayed_update,  # actor loss
                loss[1],  # qf_1 loss
                loss[2],  # qf_2 loss
                loss[3],  # vf loss
                loss[4],  # alpha loss
            )
        )

        if self.args.log:
            wandb.log(
                {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0] * delayed_update,
                    "qf_1 loss": loss[1],
                    "qf_2 loss": loss[2],
                    "vf loss": loss[3],
                    "alpha loss": loss[4],
                }
            )

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            wandb.init()
            wandb.config.update(self.hyper_params)
            wandb.watch([self.actor, self.vf, self.qf_1, self.qf_2], log="parameters")

        for i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            loss_episode = list()

            while not done:
                if self.args.render and i_episode >= self.args.render_after:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                # training
                if len(self.memory) >= self.hyper_params["BATCH_SIZE"]:
                    experiences = self.memory.sample()
                    loss = self.update_model(experiences)
                    loss_episode.append(loss)  # for logging

            # logging
            if loss_episode:
                avg_loss = np.vstack(loss_episode).mean(axis=0)
                self.write_log(
                    i_episode, avg_loss, score, self.hyper_params["DELAYED_UPDATE"]
                )

            if i_episode % self.args.save_period == 0:
                self.save_params(i_episode)

        # termination
        self.env.close()
