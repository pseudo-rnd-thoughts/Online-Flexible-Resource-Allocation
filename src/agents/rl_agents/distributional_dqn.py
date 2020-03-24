"""
Implementation of the distributional dqn from A Distributional Perspective on Reinforcement Learning (https://arxiv.org/pdf/1707.06887.pdf)
"""

import random as rnd
from abc import ABC
from typing import List, Tuple

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network
from agents.rl_agents.rl_agent import Trajectory
from env.server import Server
from env.task import Task


@gin.configurable
class DistributionalDqnAgent(DqnAgent, ABC):
    """
    Distributional Dqn Agent
    """

    def __init__(self, network: Network, num_atoms: int = 51, v_min: int = -15, v_max: int = 30, double_dqn_loss: bool = True, **kwargs):
        DqnAgent.__init__(self, network, optimiser=tf.keras.losses.CategoricalCrossentropy(), **kwargs)

        self.double_dqn_loss = double_dqn_loss

        assert v_min <= v_max
        self.num_atoms = num_atoms
        self.v_max = v_max
        self.v_min = v_min
        self.delta_z = (self.v_max - self.v_min) / self.num_atoms
        self.z = np.arange(self.v_min, self.v_max, self.delta_z)
        assert len(self.z) == self.num_atoms

    def _loss(self, trajectory: Trajectory) -> np.ndarray:
        obs = self.network_obs(trajectory.state.task, trajectory.state.tasks,
                               trajectory.state.server, trajectory.state.time_step)
        next_obs = self.network_obs(trajectory.next_state.task, trajectory.next_state.tasks,
                                    trajectory.next_state.server, trajectory.next_state.time_step)

        return np.array([])


@gin.configurable
class TaskPricingDistributionalDqnAgent(DistributionalDqnAgent, TaskPricingDqnAgent):
    """
    Distributional Dqn Task Pricing Agent
    """

    def __init__(self, network: Network, categories: int, double_dqn: bool = True, **kwargs):
        super().__init__(network, categories, double_dqn, **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if not self.eval_policy and rnd.random() < self.exploration:
            pass
        else:
            obs = self.network_obs(auction_task, allocated_tasks, server, time_step)
            action_value_prob = np.reshape(self.model_network(obs), (self.num_atoms, self.max_action_value))
            action_q = np.multiply(action_value_prob, self.z, axis=1)
            return np.argmax(action_q)


@gin.configurable
class ResourceWeightingDistributionalDqnAgent(DistributionalDqnAgent, ResourceWeightingDqnAgent):
    """
    Distributional Dqn Resource Weighting Agent
    """

    def __init__(self, network: Network, categories: int, double_dqn: bool = True, **kwargs):
        super().__init__(network, categories, double_dqn, **kwargs)

    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        return super()._get_action(weight_task, allocated_tasks, server, time_step)
