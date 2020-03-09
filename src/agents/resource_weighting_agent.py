"""
Abstract resource weighting with the abstract method _get_action function to choice how to set weights for a task

"""

from __future__ import annotations

import abc
from typing import List

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class ResourceWeightingAgent(abc.ABC):
    """
    Resource Weighting agent used in Online Flexible Resource Allocation Env in order to weight tasks
    """

    def __init__(self, name):
        self.name = name

    def weight(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        """
        Weights a task on the server with a list of already allocated tasks at time step
        Args:
            weight_task: The task being weighted
            allocated_tasks: The already allocated tasks to the server (includes the weighted task as well)
            server: The server weighting the task
            time_step: The time step of the environment

        Returns: The weight for a task

        """
        # If the length of allocated task is more than 1
        if len(allocated_tasks) > 1:
            # Assert that the task input variables are valid
            assert all(allocated_task.stage is not TaskStage.UNASSIGNED or allocated_task.stage is not TaskStage.FAILED
                       or allocated_task.stage is not TaskStage.COMPLETED for allocated_task in allocated_tasks)
            assert all(allocated_task.auction_time <= time_step <= allocated_task.deadline
                       for allocated_task in allocated_tasks)

            action = self._get_action(weight_task, allocated_tasks, server, time_step)
            assert 0 < action

            return action
        else:
            # If the weight task is only task allocated to the server
            assert weight_task == allocated_tasks[0]

            return 1.0

    @abc.abstractmethod
    def _get_action(self, weight_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        An abstract method that takes an task, a list of allocated tasks, a server and the current time step to return the weight for the task
        Args:
            weight_task: The task being weighted
            allocated_tasks: The already allocated tasks to the server (includes the weighted task as well)
            server: The server weighting the task
            time_step: The time step of the environment

        Returns: The weight for a task

        """
        pass
