"""
Defines a task pricing agent that will only bid on tasks
    where the server currently has less than fixed a number of tasks
The bid is then random for the task
"""

from __future__ import annotations

import random as rnd
from typing import List

from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


class FixedTaskPricingAgent(TaskPricingAgent):
    """
    Fixed task pricing agents
    """

    def __init__(self, agent_num: int, bid_tasks_number: int, max_price: int = 5, **kwargs):
        TaskPricingAgent.__init__(self, f'Fixed Task Pricing Agent {agent_num}', **kwargs)

        self.bid_tasks_number = bid_tasks_number
        self.max_price = max_price

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if len(allocated_tasks) <= self.bid_tasks_number:
            return float(rnd.randint(1, self.max_price))
        else:
            return 0
