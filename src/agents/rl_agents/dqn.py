"""
Deep Q Network based on Playing Atari with Deep Reinforcement Learning (https://arxiv.org/abs/1312.5602)
"""

from __future__ import annotations

from abc import ABC
from typing import List, Callable
import tensorflow as tf
import numpy as np
import random as rnd
import gin

from env.server import Server
from env.task import Task

from agents.rl_agents.rl_agent import ReinforcementLearningAgent, ResourceWeightingRLAgent, TaskPricingRLAgent


@gin.configurable
class DqnAgent(ReinforcementLearningAgent, ABC):
    """
    Deep Q Network agent
    """

    def __init__(self, network_input_width: int, network_num_outputs: int,
                 build_network: Callable[[int], tf.keras.Sequential], target_update_frequency: int = 2500,
                 initial_exploration: float = 1, final_exploration: float = 0.1, final_exploration_frame: int = 20000,
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
        ReinforcementLearningAgent.__init__(self, network_input_width, network_num_outputs, **kwargs)

        # Create the two Q network; model and target
        self.model_network = build_network(network_num_outputs)
        self.target_network = build_network(network_num_outputs)

        # The target network update frequency called from the _train function
        self.target_update_frequency = target_update_frequency

        # Exploration variables for when to choice a random action
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.exploration_gradient = (self.final_exploration - self.initial_exploration) / final_exploration_frame
        self.exploration = self.initial_exploration

        # Action selection policy
        self.greedy_policy = True

    def _train(self) -> float:
        # Get a minimatch of trajectories
        training_batch = rnd.sample(self.replay_buffer, self.batch_size)

        # The network variables to remember , the gradients and losses
        network_variables = self.model_network.trainable_variables
        gradients = []
        losses = []

        # Loop over the trajectories finding the loss and gradient
        for trajectory in training_batch:
            obs, action, reward, next_obs = trajectory

            with tf.GradientTape() as tape:
                tape.watch(network_variables)

                # Calculate the bellman update for the action
                target = np.array(self.model_network(obs))
                if next_obs is None:
                    target[0][action] = reward
                else:
                    target[0][action] = reward + np.max(self.target_network(next_obs))

                # Loss function (todo update to use the huber loss function)
                loss = tf.square(target - self.network_model(obs))

                # Add the gradient and loss to the relative lists
                gradients.append(tape.gradient(loss, network_variables))
                losses.append(tf.reduce_max(loss))

        # Calculate the mean gradient change between the losses (I believe this is equivalent to mean-square bellman error)
        mean_gradient = [np.mean(grad[var] for grad in gradients) for var in range(len(gradients[0]))]
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
class TaskPricingDQN(DqnAgent, TaskPricingRLAgent):
    """
    Task Pricing DQN agent
    """

    def __init__(self, agent_num: int, network_input_width: int, build_network: Callable[[int], tf.keras.Sequential],
                 **kwargs):
        DqnAgent.__init__(self, network_input_width, 9, build_network, **kwargs)
        TaskPricingRLAgent.__init__(self, f'DQN TP {agent_num}', network_input_width, 9)

    @staticmethod
    def network_observation(auction_task: Task, allocated_tasks: List[Task], server: Server,
                            time_step: int) -> np.ndarray:
        """
        Network observation for the Q network
        Args:
            auction_task: The task being priced
            allocated_tasks: A list of tasks already allocated to the server
            server: The server bidding on the task
            time_step: The time step of the environment

        Returns: numpy ndarray with shape (1, len(allocated_tasks) + 1, 9)

        """
        observation = np.array([
            [ReinforcementLearningAgent.normalise_task(auction_task, server, time_step) + [1.0]] +
            [ReinforcementLearningAgent.normalise_task(task, server, time_step) + [0.0] for task in allocated_tasks]
        ]).astype(np.float32)

        return observation

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if self.greedy_policy and rnd.random() < self.exploration:
            return rnd.randint(0, self.max_action_value - 1)
        else:
            action_q_value = self.network_observation(auction_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(action_q_value))


@gin.configurable
class ResourceWeightingDQN(DqnAgent, ResourceWeightingRLAgent):
    """
    Resource weighting DQN agent
    """

    @staticmethod
    def network_observation(weight_task: Task, allocated_tasks: List[Task], server: Server,
                            time_step: int) -> np.ndarray:
        """
        Network observation for the Q network
        Args:
            weight_task: The task being weighted
            allocated_tasks: The already allocated tasks to the server (includes the weighted task as well)
            server: The server weighting the task
            time_step: The time step of the environment

        Returns: numpy ndarray with shape (1, len(allocated_tasks)-1, self.max_action_value)

        """
        assert any(_task != weight_task for _task in allocated_tasks)

        task_observation = ReinforcementLearningAgent.normalise_task(weight_task, server, time_step)
        observation = np.array([[
            task_observation + ReinforcementLearningAgent.normalise_task(allocated_task, server, time_step)
            for allocated_task in allocated_tasks if allocated_tasks != allocated_task
        ]]).astype(np.float32)

        return observation

    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if self.greedy_policy and rnd.random() < self.exploration:
            return rnd.randint(1, self.max_action_value)
        else:
            action_q_values = self.network_observation(weight_task, allocated_tasks, server, time_step)
            return np.argmax(self.model_network(action_q_values)) + 1
