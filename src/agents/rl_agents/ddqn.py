"""
Double DQN agent implemented based on Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)
"""

from __future__ import annotations

import random as rnd
from abc import ABC

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.dqn import DqnAgent, TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.network import Network
from agents.rl_agents.rl_agent import Trajectory, AgentState


@gin.configurable
class DdqnAgent(DqnAgent, ABC):
    """
    Implementation of a double deep q network agent
    """

    def __init__(self, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)

    def _train(self) -> float:
        # Get a minimatch of trajectories
        training_batch = rnd.sample(self.replay_buffer, self.batch_size)

        # The network variables to remember , the gradients and losses
        network_variables = self.model_network.trainable_variables
        gradients = []
        losses = []

        # Loop over the trajectories finding the loss and gradient
        for trajectory in training_batch:
            trajectory: Trajectory

            agent_state: AgentState = trajectory.state
            action: float = trajectory.action
            reward: float = trajectory.reward
            next_agent_state: AgentState = trajectory.next_state

            with tf.GradientTape() as tape:
                tape.watch(network_variables)

                # Calculate the bellman update for the action
                obs = self.network_obs(agent_state.task, agent_state.tasks, agent_state.server, agent_state.time_step)
                target = np.array(self.model_network(obs))
                action = int(action)

                if next_agent_state is None:
                    target[0][action] = reward
                else:
                    next_obs = self.network_obs(next_agent_state.task, next_agent_state.tasks,
                                                next_agent_state.server, next_agent_state.time_step)
                    # Double Q Value modification
                    target[0][action] = reward + self.discount * self.target_network(next_obs)[np.argmax(self.model_network(next_obs))]

                if self.clip_loss:
                    loss = tf.clip_by_value(self.loss_func(target, self.model_network(obs)), -1, +1)
                else:
                    loss = self.loss_func(target, self.model_network(obs))

                # Add the gradient and loss to the relative lists
                gradients.append(tape.gradient(loss, network_variables))
                losses.append(np.max(loss))

        # Calculate the mean gradient change between the losses (therefore the mean square bellmen loss)
        mean_gradient = 1 / self.batch_size * np.mean(gradients, axis=0)

        # Apply the mean gradient to the network model
        self.optimiser.apply_gradients(zip(mean_gradient, network_variables))

        if self.total_obs % self.target_update_frequency == 0:
            self._update_target_network()
        if self.total_obs % self.exploration_frequency == 0:
            self.exploration = min(self.final_exploration,
                                   self.total_obs / self.final_exploration_frame * (
                                           self.final_exploration - self.initial_exploration) + self.initial_exploration)

        # noinspection PyTypeChecker
        return np.mean(losses)
        pass


@gin.configurable
class TaskPricingDdqnAgent(DdqnAgent, TaskPricingDqnAgent):
    """
    Task pricing double dqn agent
    """
    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        TaskPricingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'DDQN TP {agent_num}'


@gin.configurable
class ResourceWeightingDdqnAgent(DdqnAgent, ResourceWeightingDqnAgent):
    """
    Resource weighting double dqn agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DdqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingDqnAgent.__init__(self, agent_num, network, **kwargs)
        self.name = f'DDQN RW {agent_num}'
