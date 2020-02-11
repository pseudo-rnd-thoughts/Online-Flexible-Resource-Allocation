"""
Environment for online flexible resource allocation
"""

from math import inf
from random import choice
from typing import List

import core.log as log
from core.server import Server
from core.task import Task


class Environment:
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-agent mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """

    debug: bool = True

    def __init__(self, name: str, servers: List[Server], tasks: List[Task], total_time_steps: int):
        self.name: str = name

        self.servers: List[Server] = servers

        self.tasks: List[Task] = tasks
        self.unallocated_tasks: List[Task] = sorted(tasks, key=lambda task: task.auction_time)

        self.time_step: int = 0
        self.total_time_steps: int = total_time_steps

    def tick(self):
        """Simulates a time step of the environment with both stages (sub-environments) of the problem"""
        log.info('Environment - Time Step: {}'.format(self.time_step))

        # Stage 1: auction the unallocated tasks
        auction_tasks = [task for task in self.unallocated_tasks if self.time_step <= task.auction_time]

        log.info('Auction tasks')
        for task in auction_tasks:
            log.info('\t{}'.format(task))

            # Implementation of a Vickrey auction
            min_price: float = inf
            second_min_price: float = inf - 1
            min_server: List[Server] = []

            for server in self.servers:
                price = server.price_task(task, self.time_step)
                log.debug('\t\tServer {} price: {}'.format(server.name, price))

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
                task.allocate_server(server, second_min_price, self.time_step)
                self.unallocated_tasks.remove(task)

                log.info('\tMin Price: {}, Winning Server: {}, Second min price: {}'
                         .format(min_price, server.name, second_min_price))
            else:
                log.info('\tNo minimum price')

        # Stage 2: allocate the resources for each server
        log.info('\nAllocate the resources')
        for server in self.servers:
            server.allocate_resources(self.time_step)

        self.time_step += 1
