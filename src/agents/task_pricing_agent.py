"""
Abstract task pricing agent with the abstract method _get_action function to choice how to a select a price
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class TaskPricingAgent(ABC):
    """
    Task pricing agent used in Online Flexible Resource Allocation Env in order to price tasks being being auctioned
    """

    def __init__(self, name: str, limit_number_task_parallel: Optional[int] = None):
        self.name = name

        self.limit_number_task_parallel = limit_number_task_parallel

    def bid(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int) -> float:
        """
        Auctions of a task for a server with a list of already allocated tasks at time step

        Args:
            auction_task: The task being auctioned
            allocated_tasks: The already allocated tasks to the server
            server: The server bidding on the task
            time_step: The time step of the environment

        Returns: The bid value for the task

        """
        # Assert that the task input variables are valid
        assert auction_task.stage is TaskStage.UNASSIGNED
        assert auction_task.auction_time == time_step
        assert all(allocated_task.stage is not TaskStage.UNASSIGNED or allocated_task.stage is not TaskStage.FAILED or
                   allocated_task.stage is not TaskStage.COMPLETED for allocated_task in allocated_tasks)
        assert all(allocated_task.auction_time <= time_step <= allocated_task.deadline
                   for allocated_task in allocated_tasks)

        if self.limit_number_task_parallel is not None and len(allocated_tasks) < self.limit_number_task_parallel:
            action = self._get_action(auction_task, allocated_tasks, server, time_step)
            # Assert that the resulting action is valid
            assert 0 <= action

            return action
        else:
            return 0.0

    @abstractmethod
    def _get_action(self, auction_task: Task, allocated_tasks: List[Task], server: Server, time_step: int):
        """
        An abstract method that takes an auction task, a list of allocated tasks, a server
            and the current time step to return a bid price

        Args:
            auction_task: The task being auctioned
            allocated_tasks: The already allocated tasks to the server
            server: The server bidding on the task
            time_step: The time step of the environment

        Returns: The bid value for the task

        """
        pass
