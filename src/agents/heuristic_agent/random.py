"""
Agent that randoms choose the price or weight of a task
"""

import random as rnd
from typing import List

import gin
import numpy as np

from agents.resource_weighting_agents import ResourceWeightingAgent
from agents.task_pricing_agents import TaskPricingAgent


@gin.configurable
class RandomTaskPricingAgent(TaskPricingAgent):
    """
    A random task pricing agent
    """

    def __init__(self, agent_num: int, upper_price_bound: int = 10):
        TaskPricingAgent.__init__(self, f'Random TP agent {agent_num}')

        self.upper_price_bound = upper_price_bound

    def _get_action(self, task_states) -> float:
        """
        Implements the action by randomly selecting an integer price between 0 and the upper price bound

        Args:
            task_states: Ignored

        Returns: A random value between 0 and the upper price bound

        """

        return float(rnd.randint(0, self.upper_price_bound))


@gin.configurable
class RandomResourceWeightingAgent(ResourceWeightingAgent):
    """
    A random resource weighting agent
    """

    def __init__(self, agent_num: int, upper_weight_bound: int = 10):
        ResourceWeightingAgent.__init__(self, f'Random RW agent {agent_num}')

        self.upper_weight_bound = upper_weight_bound

    def _get_action(self, task_states: np.ndarray) -> List[float]:
        """
        Implements the action by randomly selecting an integer weight between 1 and the upper weight bound

        Args:
            task_states: The task states normalised by the server

        Returns: The weights for the tasks

        """
        print(f'Task states len: {len(task_states)}')
        return [float(rnd.randint(0, self.upper_weight_bound)) for _ in range(len(task_states))]
