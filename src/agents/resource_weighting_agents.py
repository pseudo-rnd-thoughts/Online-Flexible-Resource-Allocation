"""
Abstract resource weighting with the abstract method _get_action function to choice how to set weights for a task

"""

from __future__ import annotations

import abc
from typing import List
import numpy as np


class ResourceWeightingAgent(abc.ABC):
    """
    Resource Weighting agent used in Online Flexible Resource Allocation Env in order to weight tasks
    """

    def __init__(self, name):
        self.name = name

    def weight(self, task_states: np.ndarray) -> List[float]:
        """
        Weights a task on the server with a list of already allocated tasks at time step

        Args:
            task_states: The task states normalised by the server

        Returns: The weights for the tasks

        """
        if len(task_states) == 0:
            return []
        elif len(task_states) == 1:
            return [1.0]
        else:
            actions = self._get_action(task_states)
            assert all(0 <= action for action in actions)

            return actions

    @abc.abstractmethod
    def _get_action(self, task_states) -> List[float]:
        """
        An abstract method that takes an task, a list of allocated tasks, a server
            and the current time step to return the weight for the task

        Args:
            A list of normalised task states

        Returns: The weight for a task

        """
        pass
