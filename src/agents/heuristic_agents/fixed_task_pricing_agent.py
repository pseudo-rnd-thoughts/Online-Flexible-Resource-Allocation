"""
Defines a task pricing agent that will only bid on tasks
    where the server currently has less than fixed a number of tasks
The bid is then random for the task
"""

from __future__ import annotations

import random as rnd
from typing import List, Optional

from agents.rl_agents.rl_agents import TaskPricingRLAgent
from env.server import Server
from env.task import Task


class FixedTaskPricingAgent(TaskPricingRLAgent):
    """
    Fixed task pricing agents
    """

    def __init__(self, agent_num: int, bid_tasks_number: int, network_output_width: int = 5, **kwargs):
        TaskPricingRLAgent.__init__(self, f'Fixed Task Pricing Agent {agent_num}', 9, network_output_width, **kwargs)

        self.bid_tasks_number = bid_tasks_number

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        if len(allocated_tasks) <= self.bid_tasks_number:
            return float(rnd.randint(1, self.network_output_width))
        else:
            return 0

    def _train(self, states, actions, next_states, rewards, dones) -> float:
        pass

    def _save(self, custom_location: Optional[str] = None):
        pass
