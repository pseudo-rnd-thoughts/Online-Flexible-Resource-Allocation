"""
Deep Q Network based on Playing Atari with Deep Reinforcement Learning (https://arxiv.org/abs/1312.5602)
"""

from __future__ import annotations

import os
import pickle
import random as rnd
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple

import gin.tf
import numpy as np
import tensorflow as tf

from agents.rl_agents.neural_networks.network import Network
from agents.rl_agents.rl_agent import ReinforcementLearningAgent, ResourceWeightingRLAgent, TaskPricingRLAgent, \
    Trajectory, AgentState
from env.server import Server
from env.task import Task


@gin.configurable
class DqnAgent(ReinforcementLearningAgent, ABC):
    """
    Deep Q Network agent
    """

    def __init__(self, network: Network, target_update_frequency: int = 2500, initial_exploration: float = 1,
                 final_exploration: float = 0.1, final_exploration_frame: int = 100000,
                 exploration_frequency: int = 1000, discount: float = 0.9,
                 loss_func: tf.keras.losses.Loss = tf.keras.losses.Huber(), clip_loss: bool = True, **kwargs):
        """
        Constructor for the DQN agent

        Args:
            network_input_width: The network input width
            network_num_outputs: The network num of outputs
            build_network: Function to build networks
            target_update_frequency: The target network update frequency
            **kwargs: Additional arguments for the reinforcement learning agent
        """
        ReinforcementLearningAgent.__init__(self, network.input_width, network.max_action_value, **kwargs)

        # Create the two Q network; model and target
        self.model_network = network
        self.target_network = deepcopy(network)

        # The target network update frequency called from the _train function
        assert target_update_frequency % self.update_frequency == 0
        self.target_update_frequency = target_update_frequency

        # Exploration variables for when to choice a random action
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.exploration = self.initial_exploration
        self.final_exploration_frame = final_exploration_frame
        assert exploration_frequency % self.update_frequency == 0
        self.exploration_frequency = exploration_frequency

        self.loss_func = loss_func
        self.clip_loss = clip_loss

        self.discount = discount

    @staticmethod
    @abstractmethod
    def network_obs(task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> np.ndarray:
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

    def _train(self) -> float:
        # Get a minimatch of trajectories
        training_batch = rnd.sample(self.replay_buffer, self.batch_size)

        # The network variables to remember , the gradients and losses
        network_variables = self.model_network.trainable_variables
        gradients = []
        losses = []

        # Loop over the trajectories finding the loss and gradient
        for trajectory in training_batch:
            with tf.GradientTape() as tape:
                tape.watch(network_variables)

                true_target, predicted_target = self._loss(trajectory)
                if self.clip_loss:
                    loss = tf.clip_by_value(self.loss_func(true_target, predicted_target), -1, +1)
                else:
                    loss = self.loss_func(true_target, predicted_target)

                # Add the gradient and loss to the relative lists
                gradients.append(tape.gradient(loss, network_variables))
                losses.append(np.max(loss))

        # Calculate the mean gradient change between the losses (therefore the mean square bellmen loss)
        mean_gradient = 1/self.batch_size * np.mean(gradients, axis=0)

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

    def _loss(self, trajectory: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
        # Calculate the bellman update for the action
        obs = self.network_obs(trajectory.state.task, trajectory.state.tasks,
                               trajectory.state.server, trajectory.state.time_step)
        target = np.array(self.model_network(obs))
        action = int(trajectory.action)

        if trajectory.next_state is None:
            target[0][action] = trajectory.reward
        else:
            next_obs = self.network_obs(trajectory.next_state.task, trajectory.next_state.tasks,
                                        trajectory.next_state.server, trajectory.next_state.time_step)
            target[0][action] = trajectory.reward + self.discount * np.max(self.target_network(next_obs))

        return target, self.model_network(obs)

    def _update_target_network(self):
        """
        Updates the target network with the model network every target_update_frequency observations
        """
        # print(f'Update {self.name} target network {self.model_network.name}')
        self.target_network.set_weights(self.model_network.get_weights())

    def _save(self):
        path = f'{os.getcwd()}\\checkpoint\\{self.save_folder}\\{self.name.replace(" ", "_")}'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + f'\\model_{self.total_obs}.pickle', 'wb') as file:
            pickle.dump(self.model_network.trainable_variables, file)


@gin.configurable
class TaskPricingDqnAgent(DqnAgent, TaskPricingRLAgent):
    """
    Task Pricing DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)
        TaskPricingRLAgent.__init__(self, f'DQN TP {agent_num}', 9, network.max_action_value, **kwargs)

    @staticmethod
    def network_obs(pricing_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> np.ndarray:
        """
        Network observation for the Q network

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

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if not self.eval_policy and rnd.random() < self.exploration:
            return rnd.randint(0, self.max_action_value - 1)
        else:
            obs = self.network_obs(auction_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(obs))


# noinspection DuplicatedCode
@gin.configurable
class ResourceWeightingDqnAgent(DqnAgent, ResourceWeightingRLAgent):
    """
    Resource weighting DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'DQN RW {agent_num}', 10, network.max_action_value, **kwargs)

    @staticmethod
    def network_obs(weighting_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> np.ndarray:
        """
        Network observation for the Q network

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

    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if not self.eval_policy and rnd.random() < self.exploration:
            return rnd.randint(0, self.max_action_value - 1)
        else:
            obs = self.network_obs(weight_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(obs))
