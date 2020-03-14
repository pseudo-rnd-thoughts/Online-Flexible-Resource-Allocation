"""
A human agent such that for each action is decided by user input
"""

from typing import List

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


class HumanTaskPricing(TaskPricingAgent):
    """
    Human task pricing agent where a user input is required for each bid
    """

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

    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        for allocated_task in allocated_tasks:
            print(f'\t\t{str(allocated_task)}')

        weight = -1
        while weight == -1:
            try:
                weight = int(input('Enter weight: '))
                if 0 < weight:
                    print('Please enter a positive number')
                    weight = -1
            except ValueError:
                print('Please enter a number')

        return weight
