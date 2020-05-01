"""
Abstract resource weighting with the abstract method _get_action function to choice how to set weights for a task

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class ResourceWeightingAgent(ABC):
    """
    Resource Weighting agent used in Online Flexible Resource Allocation Env in order to weight tasks
    """

    def __init__(self, name):
        self.name = name

    def weight(self, allocated_tasks: List[Task], server: Server, time_step: int,
               training: bool = False) -> Dict[Task, float]:
        """
        Returns a dictionary of task with weights on server at time step

        Args:
            allocated_tasks: List of the allocated tasks on the server
            server: The server that the allocated tasks are running on
            time_step: The time step of the environment
            training: If to use training actions

        Returns: A dictionary of tasks to weights

        """
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or
                   task.stage is TaskStage.SENDING for task in allocated_tasks), \
            ', '.join([f'{task.name}: {task.stage}' for task in allocated_tasks])
        assert all(task.auction_time <= time_step <= task.deadline for task in allocated_tasks), \
            '\n'.join([str(task) for task in allocated_tasks])

        if len(allocated_tasks) <= 1:
            return {task: 1.0 for task in allocated_tasks}
        else:
            actions = self._get_actions(allocated_tasks, server, time_step, training)
            assert len(allocated_tasks) == len(actions)
            assert all(task in allocated_tasks for task in actions.keys())
            assert all(0 <= action for action in actions.values())
            assert all(type(action) is float for action in actions.values()), \
                ', '.join([str(type(action)) for action in actions.values()])

            return actions

    @abstractmethod
    def _get_actions(self, tasks: List[Task], server: Server, time_step: int,
                     training: bool = False) -> Dict[Task, float]:
        """
        An abstract method that takes a list of allocated tasks, a server and the current time of the environment
            to return a dictionary of the task to weights

        Args:
            tasks: All of the allocated tasks to the server
            server: The server running the tasks
            time_step: The time step of the environment
            training: If to use training actions

        Returns: A dictionary of tasks to weights
        """
        pass
