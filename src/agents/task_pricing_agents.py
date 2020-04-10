"""
Abstract task pricing agent with the abstract method _get_action function to choice how to a select a price
"""

from __future__ import annotations

import abc
from typing import List, Optional
import numpy as np
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


# noinspection DuplicatedCode
class TaskPricingAgent(abc.ABC):
    """
    Task pricing agent used in Online Flexible Resource Allocation Env in order to price tasks being being auctioned
    """

    def __init__(self, name, limit_number_task_parallel: Optional[int] = None):
        self.name = name

        self.limit_number_task_parallel = limit_number_task_parallel

    def bid(self, task_states: np.ndarray) -> float:
        """
        Auctions of a task for a server with a list of already allocated tasks at time step

        Args:
            task_states: Normalised list of task states

        Returns: The bid value for the task

        """

        if self.limit_number_task_parallel is not None and len(task_states) < self.limit_number_task_parallel:
            action = self._get_action(task_states)
            assert 0 <= action

            return action
        else:
            return 0.0

    @abc.abstractmethod
    def _get_action(self, task_states) -> float:
        """
        An abstract method that takes an auction task, a list of allocated tasks, a server
            and the current time step to return a bid price

        Args:
            task_states: Normalised list of task states

        Returns: The bid value for the task

        """
        pass
