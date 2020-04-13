"""
Generic Reinforcement learning agent for deep q network and policy gradient agents
"""

from __future__ import annotations

import random as rnd
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional

import gin.tf
import numpy as np
import tensorflow as tf

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


@gin.configurable
class ReinforcementLearningAgent(ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, batch_size: int = 32, optimiser: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(),
                 error_loss_fn: tf.keras.losses.Loss = tf.compat.v1.losses.huber_loss,
                 initial_training_replay_size: int = 5000, update_frequency: int = 4,
                 replay_buffer_length: int = 100000, save_frequency: int = 25000, save_folder: str = 'checkpoint',
                 **kwargs):
        """
        Todo
        Args:
            batch_size:
            optimiser:
            error_loss_fn:
            initial_training_replay_size:
            update_frequency:
            replay_buffer_length:
            save_frequency:
            save_folder:
            **kwargs:
        """
        assert 0 < batch_size
        assert 0 < update_frequency and 0 < save_frequency
        assert 0 < initial_training_replay_size and 0 < replay_buffer_length

        # Training
        self.batch_size = batch_size
        self.optimiser = optimiser
        self.error_loss_fn = error_loss_fn
        self.initial_training_replay_size = initial_training_replay_size
        self.total_updates: int = 0
        self.update_frequency = update_frequency

        # Replay buffer (todo add priority replay buffer)
        self.replay_buffer_length = replay_buffer_length
        self.replay_buffer = deque(maxlen=replay_buffer_length)
        self.total_observations: int = 0

        # Save
        self.save_frequency = save_frequency
        self.save_folder = save_folder

    @staticmethod
    def normalise_task(task: Task, server: Server, time_step: int) -> List[float]:
        """
        Normalises the task that is running on Server at environment time step

        Args:
            task: The task to be normalised
            server: The server that is the task is running on
            time_step: The current environment time step

        Returns: A list of floats where the task attributes are normalised

        """
        return [
            task.required_storage / server.storage_cap,
            task.required_storage / server.bandwidth_cap,
            task.required_comp / server.computational_comp,
            task.required_results_data / server.bandwidth_cap,
            float(task.deadline - time_step),
            task.loading_progress,
            task.compute_progress,
            task.sending_progress
        ]

    def train(self):
        """
        Trains the reinforcement learning agent and logs the training loss
        """
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.batch_size, self.replay_buffer))
        states = tf.keras.preprocessing.sequence.pad_sequences(states, dtype='float32')
        actions = tf.cast(actions, tf.float32)  # For DQN, the actions must be converted to int32
        next_states = tf.keras.preprocessing.sequence.pad_sequences(next_states, dtype='float32')
        rewards = tf.cast(rewards, tf.float32)
        dones = tf.cast(~dones, tf.int32)

        training_loss = self._train(states, actions, next_states, rewards, dones)
        tf.summary.scalar(f'{self.name} agent training loss', training_loss, step=self.total_obs)
        self.total_updates += 1

        if self.total_updates % self.save_frequency == 0:
            self._save()

    @abstractmethod
    def _train(self, states, actions, next_states, rewards, dones) -> float:
        """
        An abstract function to train the reinforcement learning agent

        Args:
            states:
            actions:
            next_states:
            rewards:
            dones:

        Returns:

        """
        pass

    @abstractmethod
    def _save(self, custom_location: Optional[str] = None):
        """
        Saves a copy of the reinforcement learning agent models at this current total obs
        """
        pass

    def _add_trajectory(self, state, action, next_state, reward, done=False):
        self.replay_buffer.append((state, action, next_state, reward, done))

        # Check if to train the agent
        self.total_observations += 1
        if self.training_replay_start_size <= self.total_obs and self.total_obs % self.update_frequency == 0:
            self.train()


@gin.configurable
class TaskPricingRLAgent(TaskPricingAgent, ReinforcementLearningAgent, ABC):
    """
    Task Pricing reinforcement learning agent
    """

    def __init__(self, name: str, failed_auction_reward: float = -0.05, failed_multiplier: float = -1.5,
                 **kwargs):
        """
        Constructor of the task pricing reinforcement learning agent

        Args:
            name: Agent name
            network_input_width: Network input width
            network_output_width: Network output width
            failed_auction_reward: Failed auction reward
            failed_reward_multiplier: Failed reward multiplier
        """
        TaskPricingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, **kwargs)

        # Reward variable
        assert failed_auction_reward <= 0, failed_auction_reward
        self.failed_auction_reward = failed_auction_reward
        assert failed_multiplier <= 0, failed_multiplier
        self.failed_multiplier = failed_multiplier

    def winning_auction_bid(self, agent_state: np.ndarray, action: float,
                            finished_task: Task, next_agent_state: np.ndarray):
        """
        When the agent is successful in winning the task then add the task when the task is finished

        Args:
            agent_state: The agent state
            action: The action
            finished_task: The finished task
            next_agent_state: The next agent state
        """
        # Check that the arguments are valid
        assert 0 <= action < self.network_output_width
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED

        # Calculate the reward and add it to the replay buffer
        reward = finished_task.price * (1 if finished_task.stage is TaskStage.COMPLETED else self.failed_multiplier)
        self._add_trajectory(agent_state, action, next_agent_state, reward)

    def failed_auction_bid(self, agent_state: np.ndarray, action: float, next_agent_state: np.ndarray):
        """
        When the agent is unsuccessful in winning the task then add the observation
            and next observation after this action

        Args:
            agent_state: The agent state
            action: The action
            next_agent_state: The next agent state
        """
        # Check that the argument are valid
        assert 0 <= action
        # assert agent_state.time_step <= next_agent_state.time_step

        # If the action is zero then there is no bid on the task so no loss
        if action == 0:
            self._add_trajectory(agent_state, action, next_agent_state, 0)
        else:
            self._add_trajectory(agent_state, action, next_agent_state, self.failed_auction_reward)


# noinspection DuplicatedCode
@gin.configurable
class ResourceWeightingRLAgent(ResourceWeightingAgent, ReinforcementLearningAgent, ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, name: str, other_task_reward_discount: float = 0.2, successful_task_reward: float = 1,
                 failed_task_reward: float = -2, task_multiplier: float = 2.0, **kwargs):
        """
        Constructor of the resource weighting reinforcement learning agent

        Args:
            name: The name of the agent
            network_input_width: The network input width
            network_output_width: The max action value
            other_task_reward_discount: The discount for when other tasks are completed
            successful_task_reward: The reward for when tasks have completed successful
            failed_task_reward: The reward for when tasks have failed
            task_multiplier: The multiplied for when the action of actual task is completed
            **kwargs: Additional arguments for the reinforcement learning agent base class
        """
        ResourceWeightingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, **kwargs)

        # Agent reward variables
        assert 0 < other_task_reward_discount
        self.other_task_reward_discount = other_task_reward_discount
        assert 0 < successful_task_reward
        self.successful_task_reward = successful_task_reward
        self.failed_task_reward = failed_task_reward
        self.task_multiplier = task_multiplier

    def allocation_obs(self, agent_state: np.ndarray, action: float, next_agent_state: np.ndarray,
                       finished_tasks: List[Task]):
        """
        Adds an observation for allocating resource but doesnt finish a task to the replay buffer

        Args:
            agent_state: The agent state
            action: The action taken
            next_agent_state: The next agent state
            finished_tasks: List of tasks that finished during that round of resource allocation
        """
        # Check that the arguments are valid
        assert 0 <= action
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)

        # If the next obs is not none or if can access empty next obs then save observation
        if next_agent_state is not None or not self.ignore_empty_next_obs:
            # Calculate the action reward
            reward = sum(self.successful_task_reward if finished_task.stage is TaskStage.COMPLETED
                         else self.failed_task_reward for finished_task in finished_tasks)
            self._add_trajectory(agent_state, action, reward, next_agent_state)

    def finished_task_obs(self, agent_state: np.ndarray, action: float, finished_task: Task,
                          finished_tasks: List[Task]):
        """
        Adds an observation for allocating resource and does finish a task
            (either successfully or unsuccessfully) to the replay buffer

        Args:
            agent_state: The agent state
            action: The action taken
            finished_task: The finished task
            finished_tasks: List of tasks that finished during that round of resource allocation
        """
        # Check that the arguments are valid
        assert 0 <= action
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)

        # Calculate the action reward
        reward = sum(
            (self.successful_task_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_task_reward) *
            (self.task_multiplier if finished_task.stage is TaskStage.COMPLETED else 1)
            for finished_task in finished_tasks)
        # Add the trajectory to the replay buffer
        self._add_trajectory(agent_state, action, reward, np.zeros(agent_state.shape), done=True)
