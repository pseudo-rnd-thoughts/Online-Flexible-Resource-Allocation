"""Resource weighting agent"""

from __future__ import annotations

import random as rnd
from typing import List

import numpy as np

import core.log as log
from agents.dqn_agent import DqnAgent
from agents.resource_weighting_network import ResourceWeightingNetwork
from env.server import Server
from env.task import Task


class ResourceWeightingAgent(DqnAgent):
    """Resource weighting agent using a resource weighting network"""

    def __init__(self, name: str, num_weights: int = 10, discount_other_task_reward: float = 0.2):
        super().__init__(name, ResourceWeightingNetwork, num_weights)
        self.discount_other_task_reward = discount_other_task_reward

    def weight_task(self, task: Task, other_tasks: List[Task], server: Server, time_step: int,
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
        observation = self.network_observation(task, other_tasks, server, time_step)

        if greedy_policy and rnd.random() < self.epsilon:
            action = rnd.randint(0, self.num_outputs)
            log.debug(f'\tGreedy action: {action}')
        else:
            action_q_values = self.network_model.call(observation)
            action = np.argmax(action_q_values) + 1
            log.debug(f'\tArgmax action: {action}')

        return action

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
