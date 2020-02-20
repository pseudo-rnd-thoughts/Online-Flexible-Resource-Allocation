"""
Environment for online flexible resource allocation
"""

from __future__ import annotations

import operator
import random as rnd
from math import inf
from typing import List, Dict, Union, Tuple, TYPE_CHECKING
from copy import deepcopy as copy

from env.env_state import EnvState
from settings import env_io

if TYPE_CHECKING:
    from env.server import Server
    from env.task import Task


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

    time_step: int = 0
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
        # regenerate the environment based on one of the random environment settings saved
        assert len(self.environment_settings) > 0

        # Select the env setting and load the environment settings
        env_setting: str = rnd.choice(self.environment_settings)
        env_name, new_servers, new_tasks, new_total_time_steps = env_io.load_setting(env_setting)

        # Update the environment variables
        self.env_setting = env_setting
        self.env_name = env_name

        self.time_step = 0  # The time step info
        self.total_time_steps = new_total_time_steps

        # Current state
        self.unallocated_tasks: List[Task] = sorted(new_tasks, key=operator.attrgetter('auction_time'))
        self.state = EnvState({server: [] for server in new_servers}, self._next_auction_task())

        return self.state

    def _next_auction_task(self):
        return self.unallocated_tasks.pop(0) if self.unallocated_tasks[0].auction_time == self.time_step else None

    def step(self, actions: Dict[Server, Union[float, Dict[Task, float]]]) -> Tuple[EnvState, Union[Dict[Server, float], Dict[Server, Dict[Task, float]]], bool, Dict[str, str]]:
        info: Dict[str, str] = {}

        # If there is an auction task then the actions must be auction
        if self.state.auction_task is not None:  # Auction action = Dict[Server, float])
            self._assert_auction_actions(actions)
            info['step type'] = 'auction'

            min_price, min_servers, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price > 0:
                    if price < min_price:
                        min_price, min_servers, second_min_price = price, [server], second_min_price
                    elif price == min_price:
                        min_servers.append(server)

            next_state: EnvState = EnvState(copy(self.state.server_tasks), self._next_auction_task())
            rewards: Dict[Server, float] = {}
            if min_servers:
                winning_server: Server = rnd.choice(min_servers)
                info['min price servers'] = f"[{', '.join(server.name for server in min_servers)}]"
                info['min price'] = str(min_price)
                info['second min price'] = str(second_min_price)
                info['winning server'] = winning_server.name

                if len(min_servers) == 1:
                    rewards[winning_server] = second_min_price
                else:
                    rewards[winning_server] = min_price
                next_state.server_tasks[winning_server].append(self.state.auction_task)
            else:
                info['min servers'] = 'failed'

        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            self._assert_resource_allocation_actions(actions)
            info['step type'] = 'resource allocation'

            next_server_tasks: Dict[Server, List[Task]] = {}
            rewards: Dict[Server, Dict[Task, float]] = {}
            for server, action_weights in actions.items():
                next_tasks, task_rewards = server.allocate_resources(action_weights)
                next_server_tasks[server] = next_tasks
                rewards[server] = task_rewards

            next_state = EnvState(next_server_tasks, self._next_auction_task())

        self.state = next_state
        return self.state, rewards, self.time_step == self.total_time_steps, info

    def _assert_auction_actions(self, actions):
        pass

    def _assert_resource_allocation_actions(self, actions):
        pass

    def __str__(self) -> str:
        if self.total_time_steps == 0:
            return 'Environment hasn\'t been generated'
        else:
            unallocated_task_str = '\n\t'.join([task.__str__() for task in self.unallocated_tasks])
            return f'{self.state.__str__()}\nUnallocated tasks\n\t{unallocated_task_str}\n'

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())