"""Task pricing agent"""

from __future__ import annotations

import random as rnd
from typing import TYPE_CHECKING

import numpy as np

import core.log as log
from agents.dqn_agent import DqnAgent
from agents.task_pricing_network import TaskPricingNetwork
from agents.trajectory import Trajectory
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.server import Server
    from env.task import Task
    from typing import List


class TaskPricingAgent(DqnAgent):
    """Task pricing agent"""

    def __init__(self, name: str, num_prices: int = 26,
                 discount_factor: float = 0.9, failed_auction_reward: float = -0.1, greedy_policy: bool = True,
                 failed_reward_multiplier: float = 1.5):
        super().__init__(name, TaskPricingNetwork, num_prices)

        self.discount_factor = discount_factor
        self.failed_auction_reward = failed_auction_reward
        self.greedy_policy = greedy_policy
        self.failed_reward_multiplier = failed_reward_multiplier

    def __str__(self) -> str:
        return f'{self.name} - Num prices: {self.num_outputs}, Discount factor: {self.discount_factor}, ' \
               f'Failed auction reward: {self.failed_auction_reward}'

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
            log.debug(f'\t{self.name} TPA - {server.name} Server has greedy action: {action}')
        else:
            action_q_values = self.network_model.call(observation)
            action = np.argmax(action_q_values)
            log.debug(f'\t{self.name} TPA - {server.name} Server has argmax action: {action}')

        return action

    @staticmethod
    def network_observation(auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> np.Array:
        observation = np.array([[auction_task.normalise(server, time_step) + [1.0]] + [
            task.normalise(server, time_step) + [0.0]
            for task in allocated_tasks
        ]]).astype(np.float32)

        return observation

    def add_finished_task(self, observation: np.Array, action: float, reward_task: Task, next_observation: np.Array):
        reward = reward_task.price if reward_task.stage is TaskStage.COMPLETED else self.failed_reward_multiplier * reward_task.price
        self.replay_buffer.append(Trajectory(observation, action, reward, next_observation))

    def add_failed_auction_task(self, observation: np.Array, action: float, next_observation: np.Array):
        if action == 0:
            self.replay_buffer.append(Trajectory(observation, action, 0, next_observation))
        else:
            self.replay_buffer.append(Trajectory(observation, action, self.failed_auction_reward, next_observation))
