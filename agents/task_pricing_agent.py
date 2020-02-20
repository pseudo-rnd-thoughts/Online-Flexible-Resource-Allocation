"""Task pricing agent"""

from __future__ import annotations

import random as rnd
from typing import List, TYPE_CHECKING

import numpy as np

import core.log as log
from agents.dqn_agent import DqnAgent
from agents.task_pricing_network import TaskPricingNetwork

if TYPE_CHECKING:
    from env.server import Server
    from env.task import Task


class TaskPricingAgent(DqnAgent):
    """Task pricing agent"""

    def __init__(self, name: str, num_prices: int = 26,
                 discount_factor: float = 0.9, default_reward: float = -0.1, greedy_policy: bool = True):
        super().__init__(name, TaskPricingNetwork, num_prices)

        self.discount_factor = discount_factor
        self.default_reward = default_reward
        self.greedy_policy = greedy_policy

    def __str__(self) -> str:
        return f'{self.name} - Num prices: {self.num_outputs}, Discount factor: {self.discount_factor}, ' \
               f'Default reward: {self.default_reward}'

    def price(self, auction_task: Task, server: Server, allocated_tasks: List[Task], time_step: int) -> float:
        """
        Get the action price for the auction task
        :param auction_task: The auction task
        :param server: The server
        :param allocated_tasks: The other allocated tasks
        :param time_step: The current time steps
        :return: the price for the task being auctioned
        """
        observation = self.network_observation(auction_task, allocated_tasks, server, time_step)

        if self.greedy_policy and rnd.random() < self.epsilon:
            action = rnd.randint(0, self.num_outputs)
            log.debug(f'\tGreedy action: {action}')
        else:
            action_q_values = self.network_model.call(observation)
            action = np.argmax(action_q_values)
            log.debug(f'\tArgmax action: {action}')

        return action

    @staticmethod
    def network_observation(auction_task: Task, allocated_tasks: List[Task], server: Server,
                            time_step: int) -> np.Array:
        observation = np.array([[auction_task.normalise(server, time_step) + [1.0]] + [
            task.normalise(server, time_step) + [0.0]
            for task in allocated_tasks
        ]]).astype(np.float32)

        return observation
