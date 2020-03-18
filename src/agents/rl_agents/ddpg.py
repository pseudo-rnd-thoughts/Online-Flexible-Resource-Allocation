"""
Deep Deterministic Policy Agent agent implemented based on Continuous control with deep reinforcement learning (https://arxiv.org/abs/1509.02971)

"""

from __future__ import annotations

import os
import pickle
import random as rnd
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network
from agents.rl_agents.rl_agent import ReinforcementLearningAgent, TaskPricingRLAgent, ResourceWeightingRLAgent, \
    AgentState, Trajectory
from env.server import Server
from env.task import Task


class Noise(ABC):
    """
    Abstract class for generating noise
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self) -> float:
        pass


class GaussianNoise(Noise):
    """
    Generates gaussian noise
    """

    def __init__(self, mean=0, std=1):
        Noise.__init__(self, 'Gaussian')

        self.mean = mean
        self.std = std

    def __call__(self) -> float:
        return rnd.gauss(self.mean, self.std)


# noinspection DuplicatedCode
@gin.configurable
class DeepDeterministicPolicyGradientAgent(ReinforcementLearningAgent, ABC):
    """
    Deep Deterministic Policy Gradient for continuous action spaces
    """

    def __init__(self, actor_network: Network, critic_network: Network, network_input_width: int,
                 max_action_value: int, tau: float, noise: Noise = GaussianNoise(), discount: float = 0.9,
                 loss_func: tf.keras.losses.Loss = tf.keras.losses.Huber(), clip_loss: bool = True):
        ReinforcementLearningAgent.__init__(self, network_input_width, max_action_value)

        self.model_actor_network = actor_network
        self.target_actor_network = deepcopy(actor_network)

        self.model_critic_network = critic_network
        self.target_critic_network = deepcopy(critic_network)

        self.tau = tau
        self.noise = noise
        self.discount = discount
        self.loss_func = loss_func
        self.clip_loss = clip_loss

    @staticmethod
    @abstractmethod
    def actor_network_obs(task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> np.ndarray:
        """
        Returns a numpy array for the network observation

        Args:
            task: The primary task to consider
            allocated_tasks: The other allocated task
            server: The server
            time_step: The time step

        Returns: numpy ndarray

        """
        pass

    @staticmethod
    @abstractmethod
    def critic_network_obs(task: Task, allocated_tasks: List[Task], server: Server,
                           time_step: int, action: float) -> np.ndarray:
        """
        Returns a numpy array for the network observation

        Args:
            task: The primary task to consider
            allocated_tasks: The other allocated task
            server: The server
            time_step: The time step
            action: The action from the actor network

        Returns: numpy ndarray
        """
        pass

    def _train(self) -> float:
        training_batch = rnd.sample(self.replay_buffer, self.batch_size)

        critic_network_variables = self.model_critic_network.trainable_variables
        critic_gradients = []
        critic_losses = []

        for trajectory in training_batch:
            trajectory: Trajectory

            agent_state: AgentState = trajectory.state
            action: float = trajectory.action
            reward: float = trajectory.reward
            next_agent_state: AgentState = trajectory.next_state

            # Find the critic loss
            with tf.GradientTape() as tape:
                tape.watch(critic_network_variables)

                critic_obs = self.critic_network_obs(agent_state.task, agent_state.tasks, agent_state.server,
                                                     agent_state.time_step, action)
                if next_agent_state is None:
                    critic_target = reward
                else:
                    actor_obs = self.actor_network_obs(next_agent_state.task, next_agent_state.tasks,
                                                       next_agent_state.server, next_agent_state.time_step)
                    critic_next_obs = self.critic_network_obs(next_agent_state.task, next_agent_state.tasks,
                                                              next_agent_state.server, next_agent_state.time_step,
                                                              self.model_actor_network(actor_obs))
                    critic_target = reward + self.discount * self.model_critic_network(critic_next_obs)

                if self.clip_loss:
                    critic_loss = tf.clip_by_value(self.loss_func(critic_target, self.model_critic_network(critic_obs)))
                else:
                    critic_loss = self.loss_func(critic_target, self.model_critic_network(critic_obs))

                critic_gradient = tape.gradient(critic_loss, critic_network_variables)
                critic_gradients.append(critic_gradient)
                critic_losses.append(critic_loss)

                # Todo add actor loss

        return np.loss(critic_losses)

    def soft_update_target_weights(self, model_network: Network, target_network: Network):
        """
        Update the target weights

        Args:
            model_network: The model network
            target_network: The target network
        """
        model_weights = model_network.get_weights()
        target_weights = target_network.get_weights()
        for pos in range(len(model_weights)):
            target_weights[pos] = self.tau * model_weights[pos] + (1 - self.tau) * target_weights[pos]
        target_network.set_weights(target_weights)

    def _save(self):
        path = f'{os.getcwd()}/checkpoint/{self.save_folder}/{self.name.replace(" ", "_")}'
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/model_{self.total_obs}_actor.pickle', 'wb') as file:
            pickle.dump(self.model_actor_network.trainable_variables, file)
        with open(f'{path}/model_{self.total_obs}_critic.pickle', 'wb') as file:
            pickle.dump(self.model_critic_network.trainable_variables, file)


# noinspection DuplicatedCode
@gin.configurable
class TaskPricingDdpgAgent(DeepDeterministicPolicyGradientAgent, TaskPricingRLAgent):
    """
    Task Pricing DDPG Agent
    """

    def __init__(self, agent_num: int, actor_network: Network, critic_network: Network,
                 network_input_width: int, max_action_value: int, **kwargs):
        DeepDeterministicPolicyGradientAgent.__init__(self, actor_network, critic_network,
                                                      network_input_width, max_action_value, **kwargs)
        TaskPricingRLAgent.__init__(self, f'DDPG TP {agent_num}', network_input_width, max_action_value, **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        obs = self.actor_network_obs(auction_task, allocated_tasks, server, time_step)
        epsilon = 0 if self.eval_policy else self.noise()
        return min(self.max_action_value, self.model_actor_network(obs) + epsilon)

    @staticmethod
    def actor_network_obs(pricing_task: Task, allocated_tasks: List[Task], server: Server,
                          time_step: int) -> np.ndarray:
        """
        Network observation for the actor network

        Args:
            pricing_task: The pricing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step

        Returns: numpy ndarray with shape (1, len(allocated_tasks) + 1, 9)
        """

        observation = np.array([
            [ReinforcementLearningAgent.normalise_task(pricing_task, server, time_step) + [1.0]] +
            [ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step) + [0.0]
             for allocated_task in allocated_tasks]
        ]).astype(np.float32)

        return observation

    @staticmethod
    def critic_network_obs(pricing_task: Task, allocated_tasks: List[Task], server: Server,
                           time_step: int, action: float) -> np.ndarray:
        """
        Network observation for the critic network

        Args:
            pricing_task: The pricing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step
            action: The action produced by the actor network

        Returns: numpy ndarray with shape (1, len(allocated_tasks) + 1, 9)
        """
        observation = np.array([
            [ReinforcementLearningAgent.normalise_task(pricing_task, server, time_step) + [
                action]] +  # Im not sure about the action being here in the observation
            [ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step) + [0.0]
             for allocated_task in allocated_tasks]
        ]).astype(np.float32)

        return observation


# noinspection DuplicatedCode
@gin.configurable
class ResourceWeightingDdpgAgent(DeepDeterministicPolicyGradientAgent, ResourceWeightingRLAgent):
    """
    Resource Weighting DDPG Agent
    """

    def __init__(self, agent_num: int, actor_network: Network, critic_network: Network,
                 network_input_width: int, max_action_value: int, **kwargs):
        DeepDeterministicPolicyGradientAgent.__init__(self, actor_network, critic_network,
                                                      network_input_width, max_action_value, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'DDPG RW {agent_num}', network_input_width, max_action_value, **kwargs)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        obs = self.actor_network_obs(auction_task, allocated_tasks, server, time_step)
        epsilon = 0 if self.eval_policy else rnd.gauss(0, self.exploration)
        return min(self.max_action_value, self.model_actor_network(obs) + epsilon)

    @staticmethod
    def actor_network_obs(weighting_task: Task, allocated_tasks: List[Task], server: Server,
                          time_step: int) -> np.ndarray:
        """
        Network observation for the actor network

        Args:
            weighting_task: The weighing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step

        Returns: numpy ndarray with shape (1, len(allocated_tasks)-1, self.max_action_value)

        """
        assert any(allocated_task != weighting_task for allocated_task in allocated_tasks)

        task_observation = ReinforcementLearningAgent.normalise_task(weighting_task, server, time_step)
        observation = np.array([[
            task_observation + ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step)
            for allocated_task in allocated_tasks if weighting_task != allocated_task
        ]]).astype(np.float32)

        return observation

    @staticmethod
    def critic_network_obs(weighting_task: Task, allocated_tasks: List[Task], server: Server,
                           time_step: int, action: float) -> np.ndarray:
        """
        Network observation for the actor network

        Args:
            weighting_task: The weighing task
            allocated_tasks: The allocated tasks
            server: The server
            time_step: The time step
            action: The action produced by the actor network

        Returns: numpy ndarray with shape (1, len(allocated_tasks)-1, self.max_action_value)

        """
        assert any(allocated_task != weighting_task for allocated_task in allocated_tasks)

        task_observation = ReinforcementLearningAgent.normalise_task(weighting_task, server, time_step)
        observation = np.array([[
            task_observation + ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step) + [action]
            for allocated_task in allocated_tasks if weighting_task != allocated_task
        ]]).astype(np.float32)

        return observation
