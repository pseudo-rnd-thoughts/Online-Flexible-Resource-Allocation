"""
Environment for online flexible resource allocation
"""

from __future__ import annotations
from math import inf
from random import choice
from typing import List, TYPE_CHECKING

import core.log as log

if TYPE_CHECKING:
    from agents.resource_weighting_agent import ResourceWeightingAgent
    from agents.task_pricing_agent import TaskPricingAgent
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

    def _repr_pretty_(self, p, cycle):
        string_tasks = '\t' + '\n\t'.join([str(task) for task in self.tasks]) + '\n'
        string_servers = '\t' + '\n'.join([str(server) for server in self.servers]) + '\n'
        p.text(f'{self.name} Environment - Time Step: {self.time_step}, Total Time Steps: {self.total_time_steps}\n'
               f'{string_tasks}\n{string_servers}')

    def set_agents(self, resource_weighting_agents: List[ResourceWeightingAgent],
                   task_pricing_agents: List[TaskPricingAgent]):
        if len(resource_weighting_agents) == len(self.servers):
            for server, resource_weighting_agent in zip(self.servers, resource_weighting_agents):
                server.resource_weighting_agent = resource_weighting_agent
        else:
            for server in self.servers:
                server.resource_weighting_agent = choice(resource_weighting_agents)

        if len(task_pricing_agents) == len(self.servers):
            for server, task_pricing_agent in zip(self.servers, task_pricing_agents):
                server.task_pricing_agent = task_pricing_agent
        else:
            for server in self.servers:
                server.task_pricing_agent = choice(task_pricing_agents)

    def run(self):
        while self.time_step < self.total_time_steps:
            self.step()

    def step(self):
        """Simulates a time step of the environment with both stages (sub-environments) of the problem"""
        log.info(f'Environment - Time Step: {self.time_step}')

        # Stage 1: auction the unallocated tasks
        auction_tasks = [task for task in self.unallocated_tasks if self.time_step <= task.auction_time]

        log.info('Auction tasks')
        for task in auction_tasks:
            log.info(f'\t{task}')

            # Implementation of a Vickrey auction
            min_price: float = inf
            second_min_price: float = inf - 1
            min_server: List[Server] = []

            for server in self.servers:
                price = server.price_task(task, self.time_step)
                log.debug(f'\t\t{server.name} Server price: {price}')

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

                log.info(
                    f'\tMin Price: {min_price}, Winning Server: {server.name}, Second min price: {second_min_price}')
            else:
                log.info('\tNo minimum price')

        # Stage 2: allocate the resources for each server
        log.info('\nAllocate the resources')
        for server in self.servers:
            server.allocate_resources(self.time_step)

        self.time_step += 1
