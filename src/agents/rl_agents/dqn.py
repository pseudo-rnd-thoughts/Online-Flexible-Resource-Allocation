"""
Deep Q Network based on Playing Atari with Deep Reinforcement Learning (https://arxiv.org/abs/1312.5602)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List
import tensorflow as tf
import numpy as np
import random as rnd
import gin

from agents.rl_agents.neural_networks.network import Network
from env.server import Server
from env.task import Task

from agents.rl_agents.rl_agent import ReinforcementLearningAgent, ResourceWeightingRLAgent, TaskPricingRLAgent, \
    Trajectory, AgentState


@gin.configurable
class DqnAgent(ReinforcementLearningAgent, ABC):
    """
    Deep Q Network agent
    """

    def __init__(self, network: Network, target_update_frequency: int = 2500,
                 initial_exploration: float = 1, final_exploration: float = 0.1, final_exploration_frame: int = 100000,
                 **kwargs):
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
        self.target_update_frequency = target_update_frequency

        # Exploration variables for when to choice a random action
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.exploration_gradient = (self.final_exploration - self.initial_exploration) / final_exploration_frame
        self.exploration = self.initial_exploration

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
                    next_obs = self.network_obs(next_agent_state.task, next_agent_state.tasks, next_agent_state.server, next_agent_state.time_step)
                    target[0][action] = reward + np.max(self.target_network(next_obs))

                # Loss function (todo update to use the huber loss function)
                loss = tf.square(target - self.model_network(obs))

                # Add the gradient and loss to the relative lists
                gradients.append(tape.gradient(loss, network_variables))
                losses.append(tf.reduce_max(loss))

        # Calculate the mean gradient change between the losses (I believe this is equivalent to mean-square bellman error)
        mean_gradient = np.mean(gradients, axis=0)

        # Apply the mean gradient to the network model
        self.optimiser.apply_gradients(zip(mean_gradient, network_variables))

        if self.total_obs % self.target_update_frequency == 0:
            self._update_target_network()
        self.exploration = min(self.final_exploration,
                               self.total_obs * self.exploration_gradient + self.initial_exploration)

        # noinspection PyTypeChecker
        return np.mean(losses)

    def _update_target_network(self):
        """
        Updates the target network with the model network every target_update_frequency observations
        """
        self.target_network.set_weights(self.model_network.get_weights())


@gin.configurable
class TaskPricingDqnAgent(DqnAgent, TaskPricingRLAgent):
    """
    Task Pricing DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)
        TaskPricingRLAgent.__init__(self, f'DQN TP {agent_num}', 9, network.max_action_value)

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
            action_q_value = self.network_obs(auction_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(action_q_value))


@gin.configurable
class ResourceWeightingDqnAgent(DqnAgent, ResourceWeightingRLAgent):
    """
    Resource weighting DQN agent
    """

    def __init__(self, agent_num: int, network: Network, **kwargs):
        DqnAgent.__init__(self, network, **kwargs)
        ResourceWeightingRLAgent.__init__(self, f'DQN RW {agent_num}', 10, network.max_action_value)

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
            return rnd.randint(1, self.max_action_value)
        else:
            action_q_values = self.network_obs(weight_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(action_q_values)) + 1
