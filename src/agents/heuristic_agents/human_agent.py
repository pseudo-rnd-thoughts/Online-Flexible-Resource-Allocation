"""
A human agent such that for each action is decided by user input
"""

from __future__ import annotations

from typing import List, Dict, Optional

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


class HumanTaskPricing(TaskPricingAgent):
    """
    Human task pricing agent where a user input is required for each bid
    """

    def __init__(self, agent_num: int, limit_number_task_parallel: Optional[int] = None):
        TaskPricingAgent.__init__(self, f'Human TP {agent_num}', limit_number_task_parallel)

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        for allocated_task in allocated_tasks:
            print(f'\t\t{str(allocated_task)}')

        price = -1
        while price == -1:
            try:
                price = int(input('Enter task bid: '))
                if 0 < price:
                    print('Please enter a positive number or zero if not bidding')
                    price = -1
            except ValueError:
                print('Please enter a number')

        return price


class HumanResourceWeighting(ResourceWeightingAgent):
    """
    Human resource weighting agent where a user input is required for each weight
    """

    def __init__(self, agent_num: int):
        ResourceWeightingAgent.__init__(self, f'Human RW {agent_num}')

    def _get_actions(self, allocated_tasks: List[Task], server: Server, time_step: int) -> Dict[Task, float]:
        task_weights = {}
        for allocated_task in allocated_tasks:
            weight = -1
            while weight == -1:
                print(f'Task: {allocated_task}')
                try:
                    weight = float(input('Enter weight: '))
                    if 0 < weight:
                        print('Please enter a positive weight')
                        weight = -1
                except ValueError:
                    print('Please enter a float number')
            task_weights[allocated_task] = weight

        return task_weights
