"""
Agent that randoms choose the price or weight of a task
"""

from __future__ import annotations

import random as rnd
from typing import List, Dict, Optional

import gin

from agents.resource_weighting_agent import ResourceWeightingAgent
from agents.rl_agents.rl_agents import TaskPricingRLAgent, ResourceWeightingRLAgent
from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task


@gin.configurable
class RandomTaskPricingAgent(TaskPricingAgent):
    """
    A random task pricing agent
    """

    def __init__(self, agent_num: int, upper_price_bound: int = 10):
        super().__init__(f'Random TP agent {agent_num}')

        self.upper_price_bound = upper_price_bound

    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int,
                    training: bool = False) -> float:
        """
        Implements the action by randomly selecting an integer price between 0 and the upper price bound

        Args:
            auction_task: Ignored
            allocated_tasks: Ignored
            server: Ignored
            time_step: Ignored

        Returns: A random value between 0 and the upper price bound

        """

        return float(rnd.randint(0, self.upper_price_bound))


@gin.configurable
class RandomResourceWeightingAgent(ResourceWeightingAgent):
    """
    A random resource weighting agent
    """

    def __init__(self, agent_num: int, upper_weight_bound: int = 10):
        super().__init__(f'Random RW agent {agent_num}')

        self.upper_weight_bound = upper_weight_bound

    def _get_actions(self, allocated_tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        """
        Implements the action by randomly selecting an integer weight between 1 and the upper weight bound

        Args:
            allocated_tasks: List of allocated tasks on the server
            server: Ignored
            time_step: Ignored

        Returns: A dictionary of tasks to weights

        """
        return {task: float(rnd.randint(0, self.upper_weight_bound)) for task in allocated_tasks}


class RandomTaskPricingRLAgent(RandomTaskPricingAgent, TaskPricingRLAgent):
    def _train(self, states, actions, next_states, rewards, dones) -> float:
        return 0

    def save(self, custom_location: Optional[str] = None):
        pass


class RandomResourceWeightingRLAgent(RandomResourceWeightingAgent, ResourceWeightingRLAgent):
    def _train(self, states, actions, next_states, rewards, dones) -> float:
        return 0

    def save(self, custom_location: Optional[str] = None):
        pass
