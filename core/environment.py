"""
Environment for online flexible resource allocation
"""

from math import inf
from random import choice
from typing import List

from core.server import Server
from core.task import Task


class Environment:
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-agent mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """

    def __init__(self, environment_setting: str, servers: List[Server], tasks: List[Task]):
        self.environment_setting: str = environment_setting
        self.servers: List[Server] = servers
        self.unallocated_tasks: List[Task] = tasks
        self.time: int = 0

    def time_step(self):
        """
        Simulates a time step of the environment with both stages (sub-environments) of the problem
        """
        # Stage 1: auction the unallocated tasks
        auction_tasks = [task for task in self.unallocated_tasks if self.time <= task.start_time]

        for task in auction_tasks:
            # Implementation of a Vickrey auction
            min_price: float = inf
            second_min_price: float = inf - 1
            min_server: List[Server] = []

            for server in self.servers:
                price = server.price_task(task)

                if price == -1:  # If a server doesnt wish to bid on a task then just returns -1
                    continue
                if price < min_price:
                    second_min_price = min_price
                    min_price = price
                    min_server = [server]
                elif price == min_price:
                    min_server.append(server)

            # If a min server is found then choice one of the minimum servers and allocate the task
            if len(min_server):
                server = choice(min_server)
                server.allocate_task(task, second_min_price)
                task.allocate_server(server, second_min_price)

                self.unallocated_tasks.remove(task)

        # Stage 2: allocate the resources for each server
        for server in self.servers:
            server.allocate_resources()

        self.time += 1
