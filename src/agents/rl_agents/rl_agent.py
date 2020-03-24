"""
Generic Reinforcement learning agent
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Optional, NamedTuple

import gin.tf
import tensorflow as tf

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class AgentState(NamedTuple):
    """
    Agent state is the information that is relevant to the agent at a particular time step for network observations
    """
    task: Task
    tasks: List[Task]
    server: Server
    time_step: int


class Trajectory(NamedTuple):
    """
    Trajectory is the information that the agent saves to replay buffer that is learnt later
    """
    state: AgentState
    action: float
    reward: float
    next_state: Optional[AgentState]


@gin.configurable
class ReinforcementLearningAgent(ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, network_input_width: int, max_action_value: float,
                 save_frequency: int = 25000, save_folder: str = 'test', batch_size: int = 32,
                 learning_rate: float = 0.001, replay_buffer_length: int = 10000,
                 update_frequency: int = 4, log_frequency: int = 80, training_replay_start_size: int = 2500, **kwargs):
        """
        Constructor of the reinforcement learning base class where the argument will be used in all subclasses

        Args:
            network_input_width: The network input width
            max_action_value: The max action width (for discrete action space, this is the network output width
                                    but for continuous action space, this is the upper action limit)
            batch_size: Training batch size
            learning_rate: Learning of the algorithm
            replay_buffer_length: The experience replay buffer length
            update_frequency: The number of observations till the agent should update
            training_replay_start_size: The minimum number of observations till the agent can start training
        """
        self.batch_size = batch_size

        # Cyclical replay buffer
        self.replay_buffer_length = replay_buffer_length
        self.replay_buffer = deque(maxlen=replay_buffer_length)

        # Adadelta optimiser and learning rate for training the agent
        self.learning_rate = learning_rate
        self.optimiser = tf.keras.optimizers.Adadelta(lr=learning_rate)

        # Network neural, input and output info
        self.network_input_width = network_input_width
        self.max_action_value = max_action_value

        # Training observations
        self.total_obs = 0
        self.training_replay_start_size = training_replay_start_size

        # Observation frequency
        assert log_frequency % update_frequency == 0, \
            f'Log Freq: {log_frequency}, Update Freq: {update_frequency}, reminder: {log_frequency % update_frequency}'
        assert save_frequency % update_frequency == 0, \
            f'Save Freq: {save_frequency}, Update Freq: {update_frequency}, reminder: {save_frequency % update_frequency}'
        self.update_frequency = update_frequency
        self.log_frequency = log_frequency
        self.save_frequency = save_frequency
        self.save_folder = save_folder

        # Policy
        self.eval_policy: bool = False

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
            task.deadline - time_step,
            task.loading_progress,
            task.compute_progress,
            task.sending_progress
        ]

    def train(self):
        """
        Trains the reinforcement learning agent and logs the training loss
        """
        training_loss = self._train()

        if self.total_obs % self.log_frequency == 0:
            tf.summary.scalar(f'{self.name} agent training loss', training_loss, step=self.total_obs)
        if self.total_obs % self.save_frequency == 0:
            self.save()

    @abstractmethod
    def _train(self) -> float:
        """
        An abstract function to train the reinforcement learning agent
        """
        pass

    def save(self):
        """
        Saves a copy of the reinforcement learning agent models at this current total obs
        """
        self._save()

    @abstractmethod
    def _save(self):
        """
        Saves a copy of the current agent neural network models
        """
        pass


# noinspection DuplicatedCode
@gin.configurable
class TaskPricingRLAgent(TaskPricingAgent, ReinforcementLearningAgent, ABC):
    """
    Task Pricing reinforcement learning agent
    """

    def __init__(self, name: str, network_input_width: int, max_action_value: float,
                 failed_auction_reward: float = -0.05, failed_reward_multiplier: float = -1.5, **kwargs):
        """
        Constructor of the task pricing reinforcement learning agent

        Args:
            name: Agent name
            network_input_width: Network input width
            max_action_value: Network output width
            failed_auction_reward: Failed auction reward
            failed_reward_multiplier: Failed reward multiplier
        """
        TaskPricingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, network_input_width, max_action_value, **kwargs)

        # Reward variable
        assert failed_auction_reward <= 0
        self.failed_auction_reward = failed_auction_reward
        assert failed_reward_multiplier <= 0
        self.failed_reward_multiplier = failed_reward_multiplier

    def winning_auction_bid(self, agent_state: AgentState, action: float,
                            finished_task: Task, next_agent_state: AgentState):
        """
        When the agent is successful in winning the task then add the task when the task is finished

        Args:
            agent_state: The agent state
            action: The action
            finished_task: The finished task
            next_agent_state: The next agent state
        """
        # Check that the arguments are valid
        assert 0 <= action < self.max_action_value
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED

        # Calculate the reward and add it to the replay buffer
        reward = finished_task.price * (
            1 if finished_task.stage is TaskStage.COMPLETED else self.failed_reward_multiplier)
        self.replay_buffer.append(Trajectory(agent_state, action, reward, next_agent_state))

        # Check if to train the agent
        self.total_obs += 1
        if self.training_replay_start_size <= self.total_obs and self.total_obs % self.update_frequency == 0:
            self.train()

    def failed_auction_bid(self, agent_state: AgentState, action: float, next_agent_state: AgentState):
        """
        When the agent is unsuccessful in winning the task then add the observation
            and next observation after this action

        Args:
            agent_state: The agent state
            action: The action
            next_agent_state: The next agent state
        """
        # Check that the argument are valid
        assert 0 <= action < self.max_action_value
        assert agent_state.time_step <= next_agent_state.time_step

        # If the action is zero then there is no bid on the task so no loss
        if action == 0:
            self.replay_buffer.append(Trajectory(agent_state, action, 0, next_agent_state))
        else:
            self.replay_buffer.append(Trajectory(agent_state, action, self.failed_auction_reward, next_agent_state))

        # Check if to train the agent
        self.total_obs += 1
        if self.training_replay_start_size <= self.total_obs and self.total_obs % self.update_frequency == 0:
            self.train()


# noinspection DuplicatedCode
@gin.configurable
class ResourceWeightingRLAgent(ResourceWeightingAgent, ReinforcementLearningAgent, ABC):
    """
    The reinforcement learning base class that is used for DQN and DDPG classes
    """

    def __init__(self, name: str, network_input_width: int, max_action_value: float,
                 other_task_reward_discount: float = 0.2, successful_task_reward: float = 1,
                 failed_task_reward: float = -2, task_multiplier: float = 2.0, ignore_empty_next_obs: bool = False,
                 **kwargs):
        """
        Constructor of the resource weighting reinforcement learning agent

        Args:
            name: The name of the agent
            network_input_width: The network input width
            max_action_value: The max action value
            other_task_reward_discount: The discount for when other tasks are completed
            successful_task_reward: The reward for when tasks have completed successful
            failed_task_reward: The reward for when tasks have failed
            task_multiplier: The multiplied for when the action of actual task is completed
            **kwargs: Additional arguments for the reinforcement learning agent base class
        """
        ResourceWeightingAgent.__init__(self, name)
        ReinforcementLearningAgent.__init__(self, network_input_width, max_action_value, **kwargs)

        # Agent reward variables
        self.other_task_reward_discount = other_task_reward_discount
        self.successful_task_reward = successful_task_reward
        self.failed_task_reward = failed_task_reward
        self.task_multiplier = task_multiplier

        # If to include the observation if the next observation is empty (i.e. it is the only task so no network observation required)
        self.ignore_empty_next_obs = ignore_empty_next_obs

    def allocation_obs(self, agent_state: AgentState, action: float,
                       next_agent_state: Optional[AgentState], finished_tasks: List[Task]):
        """
        Adds an observation for allocating resource but doesnt finish a task to the replay buffer

        Args:
            agent_state: The agent state
            action: The action taken
            next_agent_state: The next agent state
            finished_tasks: List of tasks that finished during that round of resource allocation
        """
        # Check that the arguments are valid
        assert 0 <= action < self.max_action_value
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)

        # If the next obs is not none or if can access empty next obs then save observation
        if next_agent_state is not None or not self.ignore_empty_next_obs:
            # Calculate the action reward
            reward = sum(self.successful_task_reward if finished_task.stage is TaskStage.COMPLETED
                         else self.failed_task_reward for finished_task in finished_tasks)
            self.replay_buffer.append(Trajectory(agent_state, action, reward, next_agent_state))

            # Check if to train the agent
            self.total_obs += 1
            if self.training_replay_start_size <= self.total_obs and self.total_obs % self.update_frequency == 0:
                self.train()

    def finished_task_obs(self, agent_state: AgentState, action: float, finished_task: Task,
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
        assert 0 <= action < self.max_action_value
        assert finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
        assert all(finished_task.stage is TaskStage.COMPLETED or finished_task.stage is TaskStage.FAILED
                   for finished_task in finished_tasks)

        # Calculate the action reward
        reward = sum(
            (self.successful_task_reward if finished_task.stage is TaskStage.COMPLETED else self.failed_task_reward) *
            (self.task_multiplier if finished_task.stage is TaskStage.COMPLETED else 1)
            for finished_task in finished_tasks)
        # Add the trajectory to the replay buffer
        self.replay_buffer.append(Trajectory(agent_state, action, reward, None))

        # Check if to train the agent
        self.total_obs += 1
        if self.training_replay_start_size <= self.total_obs and self.total_obs % self.update_frequency == 0:
            self.train()
