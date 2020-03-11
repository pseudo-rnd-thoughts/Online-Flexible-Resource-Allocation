"""
Agent that randoms choose the price or weight of a task
"""

from typing import List
import random as rnd

from env.server import Server
from env.task import Task
from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.task_pricing_agent import TaskPricingAgent


class RandomTaskPricingAgent(TaskPricingAgent):
    """
    A random task pricing agent
    """

    def __init__(self, agent_num: int, upper_price_bound: int = 10):
        super().__init__(f'Random TP agent {agent_num}')

        self.upper_price_bound = upper_price_bound

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        """
        Implements the action by randomly selecting an integer price between 0 and the upper price bound

        Args:
            auction_task: Ignored
            allocated_tasks: Ignored
            server: Ignored
            time_step: Ignored

        Returns: A random value between 0 and the upper price bound

        """

        return rnd.randint(0, self.upper_price_bound)


class RandomResourceWeightingAgent(ResourceWeightingAgent):
    """
    A random resource weighting agent
    """

    def __init__(self, agent_num: int, upper_weight_bound: int = 10):
        super().__init__(f'Random RW agent {agent_num}')

        self.upper_weight_bound = upper_weight_bound

    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        """
        Implements the action by randomly selecting an integer weight between 1 and the upper weight bound

        Args:
            weight_task: Ignored
            allocated_tasks: Ignored
            server: Ignored
            time_step: Ignored

        Returns: A random value between 1 and the upper weight bound

        """
        return rnd.randint(1, self.upper_weight_bound)
