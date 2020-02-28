"""Resource weighting agent"""

from __future__ import annotations

import random as rnd

from typing import TYPE_CHECKING
import numpy as np

import core.log as log
from agents.dqn_agent import DqnAgent
from agents.resource_weighting_network import ResourceWeightingNetwork
from agents.trajectory import Trajectory

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from typing import List, Optional
    from env.server import Server
    from env.task import Task


class ResourceWeightingAgent(DqnAgent):
    """Resource weighting agent using a resource weighting network"""

    def __init__(self, name: str, num_weights: int = 10, discount_other_task_reward: float = 0.2,
                 successful_task_reward: float = 1, failed_task_reward: float = -2, task_multiplier: float = 2.0):
        super().__init__(name, ResourceWeightingNetwork, num_weights)

        self.discount_other_task_reward = discount_other_task_reward
        self.successful_task_reward = successful_task_reward
        self.failed_task_reward = failed_task_reward
        self.task_multiplier = task_multiplier

    def weight(self, task: Task, other_tasks: List[Task], server: Server, time_step: int,
               greedy_policy: bool = True) -> float:
        """
        Get the action weight for the task
        :param task: The task to calculate the weight for
        :param other_tasks: The other tasks to consider
        :param server: The server of the tasks
        :param time_step: The current time step
        :param greedy_policy: If to get the policy greedly
        :return: The action weight
        """
        if other_tasks:
            observation = self.network_observation(task, other_tasks, server, time_step)

            if greedy_policy and rnd.random() < self.epsilon:
                action = rnd.randint(0, self.num_outputs)
                log.debug(f'\t{self.name} RWA - {server.name} Server and {task.name} Task has greedy action: {action}')
            else:
                action_q_values = self.network_model.call(observation)
                action = np.argmax(action_q_values) + 1
                log.debug(f'\t{self.name} TPA - {server.name} Server and {task.name} Task has argmax action: {action}')

            return action
        else:
            return 1.0

    @staticmethod
    def network_observation(task: Task, other_tasks: List[Task], server: Server, time_step: int):
        """
        The network observation
        :param task: The weighting task
        :param other_tasks: The other tasks
        :param server: The allocated server
        :param time_step: The current time step
        :return: Network observation
        """
        task_observation = task.normalise(server, time_step)
        observation = np.array([[
            task_observation + task.normalise(server, time_step)
            for task in other_tasks
        ]]).astype(np.float32)

        return observation

    def add_incomplete_task_observation(self, observation: np.Array, action: float,
                                        next_observation: Optional[np.Array], rewards: List[Task]):
        reward = sum(
            self.successful_task_reward if reward_task.stage is TaskStage.COMPLETED else self.failed_task_reward
            for reward_task in rewards)

        self.replay_buffer.append(Trajectory(observation, action, reward, next_observation))

    def add_finished_task(self, observation: np.Array, action: float, finished_task: Optional[np.Array],
                          rewards: List[Task]):
        reward = self.successful_task_reward * self.task_multiplier if finished_task.stage is TaskStage.COMPLETED else \
            self.failed_task_reward * self.task_multiplier
        for reward_task in rewards:
            if reward_task.name != finished_task.name:
                reward += self.successful_task_reward if reward_task.stage is TaskStage.COMPLETED else self.failed_task_reward

        self.replay_buffer.append(Trajectory(observation, action, reward, None))
