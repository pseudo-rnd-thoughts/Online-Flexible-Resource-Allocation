"""
Defines a task pricing agent that will only bid on tasks
    where the server currently has less than fixed a number of tasks
The bid is then random for the task
"""

import random as rnd
from typing import List

from agents.rl_agents.rl_agent import TaskPricingRLAgent
from env.server import Server
from env.task import Task


class FixedTaskPricingAgent(TaskPricingRLAgent):
    """
    Fixed task pricing agents
    """

    def __init__(self, agent_num: int, bid_tasks_number: int, max_action_value: int = 5, **kwargs):
        TaskPricingRLAgent.__init__(self, f'{agent_num} Fixed Task Pricing Agent', 9, max_action_value, **kwargs)

        self.bid_tasks_number = bid_tasks_number

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if len(allocated_tasks) <= self.bid_tasks_number:
            return rnd.randint(1, self.max_action_value)
        else:
            return 0

    def _train(self) -> float:
        return 0

    def _save(self):
        pass
