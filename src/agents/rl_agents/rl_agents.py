"""
Generic Reinforcement learning agent for deep q network and policy gradient agents
"""

from __future__ import annotations

import random as rnd
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, NamedTuple, Dict

import gin.tf
import numpy as np
import tensorflow as tf

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class TaskPricingState(NamedTuple):
    """
    Task pricing reinforcement learning agent state
    """
    auction_task: Task
    tasks: List[Task]
    server: Server
    time_step: int


class ResourceAllocationState(NamedTuple):
    """
    Resource allocation reinforcement learning agent state
    """
    tasks: List[Task]
    server: Server
    time_step: int


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
        Constructor that is generalised for the deep q networks and policy gradient agents
        Args:
            batch_size: Training batch sizes
            optimiser: Network optimiser
            error_loss_fn: Training error loss function
            initial_training_replay_size: The required initial training replay size
            update_frequency: Network update frequency
            replay_buffer_length: Replay buffer length
            save_frequency: Agent save frequency
            save_folder: Agent save folder
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
    def _normalise_task(task: Task, server: Server, time_step: int) -> List[float]:
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
            task.required_computation / server.computational_cap,
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
        states, actions, next_states, rewards, dones = zip(*rnd.sample(self.replay_buffer, self.batch_size))

        states = tf.keras.preprocessing.sequence.pad_sequences(list(states), dtype='float32')
        actions = tf.cast(tf.stack(actions), tf.float32)  # For DQN, the actions must be converted to int32
        next_states = tf.keras.preprocessing.sequence.pad_sequences(list(next_states), dtype='float32')
        rewards = tf.cast(tf.stack(rewards), tf.float32)
        dones = tf.cast(tf.stack(dones), tf.float32)

        training_loss = self._train(states, actions, next_states, rewards, dones)
        tf.summary.scalar(f'{self.name} agent training loss', training_loss, step=self.total_observations)
        self.total_updates += 1

        if self.total_updates % self.save_frequency == 0:
            self._save()

    @abstractmethod
    def _train(self, states, actions, next_states, rewards, dones) -> float:
        """
        An abstract function to train the reinforcement learning agent

        Args:
            states: Tensor of network observations
            actions: Tensor of actions
            next_states: Tensor of the next network observations
            rewards: Tensor of rewards
            dones: Tensor of dones

        Returns: Training loss

        """
        pass

    @abstractmethod
    def _save(self, custom_location: Optional[str] = None):
        """
        Saves a copy of the reinforcement learning agent models at this current total obs
        """
        pass

    def _add_trajectory(self, state, action: float, next_state, reward: float, done: bool = False):
        if done:
            self.replay_buffer.append((state, action, next_state, reward, 0))
        else:
            self.replay_buffer.append((state, action, next_state, reward, 1))

        # Check if to train the agent
        self.total_observations += 1
        if self.initial_training_replay_size <= self.total_observations and \
                self.total_observations % self.update_frequency == 0:
            self.train()


@gin.configurable
class TaskPricingRLAgent(TaskPricingAgent, ReinforcementLearningAgent, ABC):
    """
    Task Pricing reinforcement learning agent
    """

    def __init__(self, name: str, failed_auction_reward: float = -0.05, failed_multiplier: float = -1.5, **kwargs):
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

    @staticmethod
    @abstractmethod
    def _network_obs(task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        Returns a list for the network observation

        Args:
            task: The primary task to consider
            allocated_tasks: The other allocated task
            server: The server
            time_step: The time step

        Returns: List for the network observation
        """
        pass

    def winning_auction_bid(self, agent_state: TaskPricingState, action: float,
                            finished_task: Task, next_agent_state: TaskPricingState):
        """
        When the agent is successful in winning the task then add the task when the task is finished

        Args:
            agent_state: Initial agent state
            action: Auction action
            finished_task: Auctioned finished task containing the winning price
            next_agent_state: Resulting next agent state
        """
        # Check that the arguments are valid
        assert 0 <= action
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED, finished_task

        # Calculate the reward and add it to the replay buffer
        reward = finished_task.price * (1 if finished_task.stage is TaskStage.COMPLETED else self.failed_multiplier)
        obs = self._network_obs(agent_state.auction_task, agent_state.tasks, agent_state.server, agent_state.time_step)
        next_obs = self._network_obs(next_agent_state.auction_task, next_agent_state.tasks,
                                     next_agent_state.server, next_agent_state.time_step)

        self._add_trajectory(obs, action, next_obs, reward)

    def failed_auction_bid(self, agent_state: TaskPricingState, action: float, next_agent_state: TaskPricingState):
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
        obs = self._network_obs(agent_state.auction_task, agent_state.tasks, agent_state.server, agent_state.time_step)
        next_obs = self._network_obs(next_agent_state.auction_task, next_agent_state.tasks,
                                     next_agent_state.server, next_agent_state.time_step)
        self._add_trajectory(obs, action, next_obs, self.failed_auction_reward if action == 0 else 0)


@gin.configurable
class ResourceWeightingRLAgent(ResourceWeightingAgent, ReinforcementLearningAgent, ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, name: str, other_task_discount: float = 0.2, success_reward: float = 1,
                 failed_reward: float = -2, reward_multiplier: float = 2.0,
                 ignore_empty_next_obs: bool = True, **kwargs):
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
        assert 0 < other_task_discount
        self.other_task_discount = other_task_discount
        assert 0 < success_reward
        self.success_reward = success_reward
        self.failed_reward = failed_reward
        self.reward_multiplier = reward_multiplier
        self.ignore_empty_next_obs = ignore_empty_next_obs

    @staticmethod
    @abstractmethod
    def _network_obs(task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        Returns a numpy array for the network observation

        Args:
            task: The primary task to consider
            allocated_tasks: The other allocated task
            server: The server
            time_step: The time step

        Returns: numpy ndarray for the network observation
        """
        pass

    def resource_allocation_obs(self, agent_state: ResourceAllocationState, actions: Dict[Task, float],
                                next_agent_state: ResourceAllocationState, finished_tasks: List[Task]):
        """
        Adds a resource allocation state and actions with the resulting resource allocation state with the list of
            finished tasks

        Args:
            agent_state: Resource allocation state
            actions: List of actions
            next_agent_state: Next resource allocation state
            finished_tasks: List of tasks that finished during that round of resource allocation
        """
        # Check that the arguments are valid
        assert len(agent_state.tasks) == len(actions)
        assert all(task in agent_state.tasks for task in actions.keys())
        assert all(0 <= action for action in actions.values())
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)
        assert all(task in next_agent_state.tasks or task in finished_tasks for task in agent_state.tasks)

        if len(agent_state.tasks) <= 1 or (len(next_agent_state.tasks) <= 1 and self.ignore_empty_next_obs):
            return

        for task, action in actions.items():
            obs = self._network_obs(task, agent_state.tasks, agent_state.server, agent_state.time_step)
            reward = sum(self.success_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_reward
                         for finished_task in finished_tasks if not task == finished_task) * self.other_task_discount
            if task in next_agent_state.tasks:
                if 1 < len(next_agent_state.tasks):
                    next_task = next(next_task for next_task in next_agent_state.tasks if next_task == task)
                    next_obs = self._network_obs(next_task, next_agent_state.tasks, next_agent_state.server, next_agent_state.time_step)
                    self._add_trajectory(obs, action, next_obs, reward)
            else:
                next_obs = np.zeros((1, self.resource_obs_width))
                finished_task = next(finished_task for finished_task in finished_tasks if finished_task == task)
                reward += (self.success_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_reward) * self.reward_multiplier

                self._add_trajectory(obs, action, next_obs, reward, done=True)

    """
    if len(tasks) > 1:
    # Get the agent state for each task
    for weighted_task in tasks:
        # Get the last agent state that generated the weighting
        last_agent_state = ResourceAllocationState(weighted_task, tasks, server, state.time_step)
        last_action = resource_weighting_actions[server][weighted_task]

        # Get the modified task in the next state, the task may be missing if the task is finished
        updated_task = next((next_task for next_task in next_state.server_tasks[server]
                             if weighted_task == next_task), None)

        # If the task wasn't finished
        if updated_task:
            # Check if the next state contains other tasks than the updated task
            if len(next_state.server_tasks[server]) > 1:
                # Get the next observation (imagining that no new tasks were auctioned)
                next_agent_state = AgentState(updated_task, next_state.server_tasks[server], server,
                                              next_state.time_step)

                # Add the task observation with the rewards of other tasks completed
                server_resource_weighting_agents[server].allocation_obs(last_agent_state, last_action,
                                                                        next_agent_state,
                                                                        finished_server_tasks[server])
            else:
                # Add the task observation but without the next observations
                server_resource_weighting_agents[server].allocation_obs(last_agent_state, last_action, None,
                                                                        finished_server_tasks[server])
        else:
            # The weighted task was finished so using the finished task in the finished_server_tasks dictionary
            finished_task = next(finished_task for finished_task in finished_server_tasks[server]
                                 if finished_task == weighted_task)

            # Update the resource allocation with teh finished task observation
            server_resource_weighting_agents[server].finished_task_obs(last_agent_state, last_action,
                                                                       finished_task,
                                                                       finished_server_tasks[server])
    """
