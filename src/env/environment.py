"""
Environment for online flexible resource allocation
"""

from __future__ import annotations

import operator
import random as rnd
from math import inf
from typing import TYPE_CHECKING, Optional
from copy import deepcopy as copy

from env.env_state import EnvState
from env.task_stage import TaskStage
import settings.env_io as env_io

if TYPE_CHECKING:
    from env.server import Server
    from env.task import Task
    from typing import List, Dict, Union, Tuple


class OnlineFlexibleResourceAllocationEnv:
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-agent mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """

    env_setting: str = ''
    env_name: str = ''

    unallocated_tasks: List[Task]
    state: EnvState

    total_time_steps: int = 0

    def __init__(self, environment_settings: List[str]):
        self.environment_settings: List[str] = environment_settings

    @staticmethod
    def make(settings: Union[List[str], str]) -> OnlineFlexibleResourceAllocationEnv:
        """
        Creates the environment using the provided settings
        :param settings: The available settings
        :return: A new OnlineFlexibleResourceAllocation environment
        """
        if type(settings) is list:
            assert len(settings) > 0
            assert all(type(setting) is str for setting in settings)

            return OnlineFlexibleResourceAllocationEnv(settings)
        elif type(settings) is str:
            return OnlineFlexibleResourceAllocationEnv([settings])

    def reset(self) -> EnvState:
        """
        Resets the environment using one of the environment settings that is randomly chosen
        :return: The new environment state
        """
        assert len(self.environment_settings) > 0

        # Select the env setting and load the environment settings
        env_setting: str = rnd.choice(self.environment_settings)
        env_name, new_servers, new_tasks, new_total_time_steps = env_io.load_setting(env_setting)

        assert len(new_tasks) > 0
        assert len(new_servers) > 0

        # Update the environment variables
        self.env_setting = env_setting
        self.env_name = env_name

        # Current state
        self.total_time_steps = new_total_time_steps
        self.unallocated_tasks: List[Task] = sorted(new_tasks, key=operator.attrgetter('auction_time'))
        auction_task = self.unallocated_tasks.pop(0) if self.unallocated_tasks[0].auction_time == 0 else None
        self.state = EnvState({server: [] for server in new_servers}, auction_task, 0)

        return self.state

    def next_auction_task(self, time_step: int) -> Optional[Task]:
        """
        Gets the next auction task if a task with auction time == current time step exists in the unallocated tasks
        :return: The auction task
        """
        assert time_step >= 0
        if self.unallocated_tasks:
            assert self.unallocated_tasks[0].auction_time >= time_step, \
                f'Top unallocated task auction time {self.unallocated_tasks[0].auction_time} at time step: {time_step}'
            return self.unallocated_tasks.pop(0) if self.unallocated_tasks[0].auction_time == time_step else None

    def step(self, actions: Dict[Server, Union[float, Dict[Task, float]]]) -> Tuple[EnvState, Dict[Server, Union[float, List[Task]]], bool, Dict[str, str]]:
        """
        An environment step that is either an auction step or a resource allocation step
        :param actions: The actions can be for auction or resource allocation meaning the data structure changes
        :return: A tuple of environment state, rewards, if done and information
        """
        info: Dict[str, str] = {}

        # If there is an auction task then the actions must be auction
        if self.state.auction_task is not None:  # Auction action = Dict[Server, float])
            info['step type'] = 'auction'

            # Vickrey auction, the server wins with the minimum price but only pays the second minimum price
            #  If multiple servers all price the same price then the server pays the minimum price (not second minimum price)
            min_price, min_servers, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price > 0:  # If the price is zero, then the bid is ignored
                    if price < min_price:
                        min_price, min_servers, second_min_price = price, [server], second_min_price
                    elif price == min_price:
                        min_servers.append(server)

            # Creates the next environment state by copying the server task info, get the next auction task and the time step doesnt change
            next_state: EnvState = EnvState(copy(self.state.server_tasks),
                                            self.next_auction_task(self.state.time_step),
                                            self.state.time_step)
            # The reward dictionary of server to price (this is only for the server that won)
            rewards: Dict[Server, float] = {}

            # Select the winning server and update the next state with the auction task
            if min_servers:
                winning_server: Server = rnd.choice(min_servers)
                info['min price servers'] = f"[{', '.join(server.name for server in min_servers)}]"
                info['min price'] = str(min_price)
                info['second min price'] = str(second_min_price)
                info['winning server'] = winning_server.name

                # Update the next state servers with the auction task
                if min_servers:
                    price = second_min_price if len(min_servers) == 1 and second_min_price < inf else min_price
                    rewards[winning_server] = price
                    next_state.server_tasks[winning_server].append(self.state.auction_task._replace(stage=TaskStage.LOADING, price=price))
            else:
                info['min servers'] = 'failed, no server won'

        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            info['step type'] = 'resource allocation'

            # The updated server tasks and the resulting rewards
            next_server_tasks: Dict[Server, List[Task]] = {}
            rewards: Dict[Server, List[Task]] = {}

            # For each server, if the server has tasks then allocate resources using the task weights
            for server, task_resource_weights in actions.items():
                if self.state.server_tasks[server]:
                    # Allocate resources returns two lists, one of unfinished tasks and the other of finished tasks
                    unfinished_tasks, completed_tasks = server.allocate_resources(task_resource_weights, self.state.time_step)
                    next_server_tasks[server] = unfinished_tasks
                    rewards[server] = completed_tasks
                else:
                    next_server_tasks[server] = []

            # The updated state
            next_state = EnvState(next_server_tasks,
                                  self.next_auction_task(self.state.time_step + 1),
                                  self.state.time_step + 1)

        # Update the state, and return the next state, the action rewards, if done and any additional info
        assert all(server in next_state.server_tasks.keys() for server in self.state.server_tasks.keys())
        assert all(id(task) != id(_task) for server, state_tasks in self.state.server_tasks.items()
                   for task in state_tasks for _task in next_state.server_tasks[server])
        self.state = next_state
        return self.state, rewards, self.total_time_steps < self.state.time_step, info

    def __str__(self) -> str:
        if self.total_time_steps == 0:
            return 'Environment hasn\'t been generated'
        else:
            unallocated_task_str = '\n\t'.join([task.__str__() for task in self.unallocated_tasks])
            return f'{self.state.__str__()}\nUnallocated tasks\n\t{unallocated_task_str}\n'

    # noinspection PyUnusedLocal
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
