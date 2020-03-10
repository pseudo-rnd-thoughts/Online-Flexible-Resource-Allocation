"""
Environment for online flexible resource allocation
"""

from __future__ import annotations

import json
import operator
import random as rnd
from math import inf
from typing import TYPE_CHECKING, Optional
from copy import deepcopy as copy

from env.env_state import EnvState
from env.task_stage import TaskStage
from env.server import Server
from env.task import Task

if TYPE_CHECKING:
    from typing import List, Dict, Union, Tuple

    ACTION_TYPE = Dict[Server, Union[float, Dict[Task, float]]]
    REWARD_TYPE = Dict[Server, Union[float, List[Task]]]


class OnlineFlexibleResourceAllocationEnv:
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-rl_agents mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """

    def __init__(self, environment_settings: List[str]):
        self._environment_settings: List[str] = environment_settings

        self._env_setting: str = ''
        self._env_name: str = ''

        self._unallocated_tasks: List[Task] = []
        self._state: EnvState = EnvState({}, None, -1)

        self._total_time_steps: int = 0

    @staticmethod
    def make(settings: Union[List[str], str]) -> OnlineFlexibleResourceAllocationEnv:
        """
        Creates the environment using the provided env_settings

        Args:
            settings: The settings to which to load the environment from

        Returns:  A new OnlineFlexibleResourceAllocation environment

        """
        if type(settings) is list:
            assert len(settings) > 0
            assert all(type(setting) is str for setting in settings)

            return OnlineFlexibleResourceAllocationEnv(settings)
        elif type(settings) is str:
            return OnlineFlexibleResourceAllocationEnv([settings])

    @staticmethod
    def custom_env(env_name: str, total_time_steps: int, new_servers_tasks: Dict[Server, List[Task]],
                   new_unallocated_tasks: List[Task]) -> Tuple[OnlineFlexibleResourceAllocationEnv, EnvState]:
        """
        Setup a custom environment

        Args:
            env_name: The environment name
            total_time_steps: The total time steps of the environment
            new_servers_tasks: A dictionary of server to list of tasks
            new_unallocated_tasks: A list of unallocated tasks

        Returns: A tuple of new environment and its state

        """
        env = OnlineFlexibleResourceAllocationEnv.make('custom')
        env._env_name = env_name

        assert 0 < total_time_steps
        assert 0 < len(new_servers_tasks)
        assert all(task.stage is not TaskStage.UNASSIGNED or task.stage is not TaskStage.COMPLETED or task.stage is not TaskStage.FAILED
                   for _, tasks in new_servers_tasks.items() for task in tasks)
        assert all(task.stage is TaskStage.UNASSIGNED for task in new_unallocated_tasks)

        env._total_time_steps = total_time_steps
        env._unallocated_tasks = sorted(new_unallocated_tasks, key=operator.attrgetter('auction_time'))
        auction_task = env._unallocated_tasks.pop(0) if env._unallocated_tasks[0].auction_time == 0 else None
        env._state = EnvState(new_servers_tasks, auction_task, 0)

        return env, env._state

    def reset(self) -> EnvState:
        """
        Resets the environment using one of the environment env_settings that is randomly chosen

        Returns: The new environment state

        """
        assert len(self._environment_settings) > 0

        # Select the env setting and load the environment env_settings
        env_setting: str = rnd.choice(self._environment_settings)
        env_name, new_servers, new_tasks, new_total_time_steps = self._load_setting(env_setting)

        assert len(new_tasks) > 0
        assert len(new_servers) > 0

        # Update the environment variables
        self._env_setting = env_setting
        self._env_name = env_name

        # Current state
        self._total_time_steps = new_total_time_steps
        self._unallocated_tasks: List[Task] = sorted(new_tasks, key=operator.attrgetter('auction_time'))
        auction_task = self._unallocated_tasks.pop(0) if self._unallocated_tasks[0].auction_time == 0 else None
        self._state = EnvState({server: [] for server in new_servers}, auction_task, 0)

        return self._state

    def _next_auction_task(self, time_step: int) -> Optional[Task]:
        """
        Gets the next auction task if a task with auction time == current time step exists in the unallocated tasks

        Args:
            time_step: The time step that the task auction time must be time step

        Returns: the new auction task (none if no task to auction)

        """
        assert time_step >= 0
        if self._unallocated_tasks:
            assert self._unallocated_tasks[0].auction_time >= time_step, \
                f'Top unallocated task auction time {self._unallocated_tasks[0].auction_time} at time step: {time_step}'
            return self._unallocated_tasks.pop(0) if self._unallocated_tasks[0].auction_time == time_step else None

    def step(self, actions: ACTION_TYPE) -> Tuple[EnvState, REWARD_TYPE, bool, Dict[str, str]]:
        """
        An environment step that is either an auction step or a resource allocation step

        Args:
            actions: The actions can be for auction or resource allocation meaning the data structure changes

        Returns:A tuple of environment state, rewards, if done and information

        """
        info: Dict[str, str] = {}

        # If there is an auction task then the actions must be auction
        if self._state.auction_task is not None:  # Auction action = Dict[Server, float])
            info['step type'] = 'auction'
            assert all(type(actions[server]) is float and 0 <= actions[server]
                       for server in self._state.server_tasks.keys())

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
            next_state: EnvState = EnvState(copy(self._state.server_tasks),
                                            self._next_auction_task(self._state.time_step),
                                            self._state.time_step)
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
                    updated_task = self._state.auction_task.assign(price, self._state.time_step)
                    next_state.server_tasks[winning_server].append(updated_task)
            else:
                info['min servers'] = 'failed, no server won'

        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            info['step type'] = 'resource allocation'
            assert all(type(actions[server][task]) is float and 0 < actions[server][task]
                       for server, tasks in self._state.server_tasks.items() for task in tasks)

            # The updated server tasks and the resulting rewards
            next_server_tasks: Dict[Server, List[Task]] = {}
            rewards: Dict[Server, List[Task]] = {}

            # For each server, if the server has tasks then allocate resources using the task weights
            for server, task_resource_weights in actions.items():
                if self._state.server_tasks[server]:
                    # Allocate resources returns two lists, one of unfinished tasks and the other of finished tasks
                    next_server_tasks[server], rewards[server] = server.allocate_resources(task_resource_weights,
                                                                                           self._state.time_step)
                else:
                    next_server_tasks[server] = []

            # The updated state
            next_state = EnvState(next_server_tasks,
                                  self._next_auction_task(self._state.time_step + 1),
                                  self._state.time_step + 1)

        # Update the state, and return the next state, the action rewards, if done and any additional info
        assert all(server in next_state.server_tasks.keys() for server in self._state.server_tasks.keys())
        assert all(id(task) != id(_task) for server, state_tasks in self._state.server_tasks.items()
                   for task in state_tasks for _task in next_state.server_tasks[server])

        self._state = next_state
        return self._state, rewards, self._total_time_steps < self._state.time_step, info

    def __str__(self) -> str:
        if self._total_time_steps == 0:
            return 'Environment hasn\'t been generated'
        else:
            unallocated_task_str = '\n\t'.join([task.__str__() for task in self._unallocated_tasks])
            return f'{self._state.__str__()}\nUnallocated tasks\n\t{unallocated_task_str}\n'

    # noinspection PyUnusedLocal
    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def save(self, filename: str):
        """
        Saves this environment to a file with the following template
        {"name": "",
         "total time steps": 0,
         "servers": [{"name": "", "storage capacity": 0, "computational capacity": 0, "bandwidth capacity": 0}, ...],
         "tasks": [{"name": "", "required storage": 0, "required computation": 0, "required results data": 0,
                    "auction time": 0, "deadline": 0}, ...]
        }

        Args:
            filename: The filename to save the environment to

        """
        assert self._state.time_step == 0

        environment_json_data = {
            'env name': self._env_name,
            'total time steps': self._total_time_steps,
            'servers': [
                {'name': server.name, 'storage capacity': server.storage_cap,
                 'computational capacity': server.computational_comp,
                 'bandwidth capacity': server.bandwidth_cap}
                for server in self._state.server_tasks.keys()
            ],
            'tasks': [
                {'name': task.name, 'required storage': task.required_storage,
                 'required computational': task.required_comp, 'required results data': task.required_results_data,
                 'auction time': task.auction_time, 'deadline': task.deadline}
                for task in self._unallocated_tasks
            ]
        }

        with open(filename, 'w') as file:
            json.dump(environment_json_data, file)

    @staticmethod
    def load(filename: str) -> Tuple[OnlineFlexibleResourceAllocationEnv, EnvState]:
        """
        Loads an environment from a file from template
        {"name": "",
         "total time steps": 0,
         "servers": [{"name": "", "storage capacity": 0, "computational capacity": 0, "bandwidth capacity": 0}, ...],
         "tasks": [{"name": "", "required storage": 0, "required computation": 0, "required results data": 0,
                    "auction time": 0, "deadline": 0}, ...]
        }

        Args:
            filename: The filename to load the environment from

        Returns: The loaded environment

        """

        with open(filename) as file:
            json_data = json.load(file)

            name: str = json_data['env name']
            total_time_steps: int = json_data['total time steps']

            # Load the servers list
            servers: List[Server] = [
                Server(name=server_data['name'], storage_cap=server_data['storage capacity'],
                       computational_comp=server_data['computational capacity'],
                       bandwidth_cap=server_data['bandwidth capacity'])
                for server_data in json_data['servers']
            ]

            # Load the tasks list
            tasks: List[Task] = [
                Task(name=task_data['name'], auction_time=task_data['auction time'], deadline=task_data['deadline'],
                     required_storage=task_data['required storage'], required_comp=task_data['required computational'],
                     required_results_data=task_data['required results data'])
                for task_data in json_data['tasks']
            ]

        env = OnlineFlexibleResourceAllocationEnv([filename])
        env._env_name = name
        env._total_time_steps = total_time_steps
        env._unallocated_tasks = sorted(tasks, key=operator.attrgetter('auction_time'))
        new_state = EnvState({server: [] for server in servers}, env._next_auction_task(0), 0)
        env._state = new_state
        return env, new_state

    @staticmethod
    def _load_setting(filename: str) -> Tuple[str, List[Server], List[Task], int]:
        """
        Load an environment env_settings from a file with a number of environments with the following template
        {"name": "",
         "min total time steps": 0, "max total time steps": 0,
         "min total servers": 0, "max total servers": 0,
         "server env_settings": [
            {
              "name": "", "min storage capacity": 0, "max storage capacity": 0,
              "min computational capacity": 0, "max computational capacity": 0,
              "min bandwidth capacity": 0, "max bandwidth capacity": 0
            }, ...],
         "task env_settings": [
            {
              "name": "", "min deadline": 0, "max deadline": 0,
              "min required storage": 0, "max required storage": 0,
              "min required computation": 0, "max required computation": 0,
              "min required results data": 0, "max required results data": 0
            }, ...]
         }

        Args:
            filename: The filename to loads the env_settings from

        Returns: Returns the primary features of an environment to be set

        """

        with open(filename) as file:
            env_setting_json = json.load(file)

            env_name = env_setting_json['name']
            total_time_steps = rnd.randint(env_setting_json['min total time steps'],
                                           env_setting_json['max total time steps'])
            servers: List[Server] = []
            for server_num in range(rnd.randint(env_setting_json['min total servers'],
                                                env_setting_json['max total servers'])):
                server_json_data = rnd.choice(env_setting_json['server settings'])
                servers.append(Server(
                    name='{} {}'.format(server_json_data['name'], server_num),
                    storage_cap=float(rnd.randint(server_json_data['min storage capacity'],
                                                  server_json_data['max storage capacity'])),
                    computational_comp=float(rnd.randint(server_json_data['min computational capacity'],
                                                         server_json_data['max computational capacity'])),
                    bandwidth_cap=float(rnd.randint(server_json_data['min bandwidth capacity'],
                                                    server_json_data['max bandwidth capacity']))))

            tasks: List[Task] = []
            for task_num in range(rnd.randint(env_setting_json['min total tasks'],
                                              env_setting_json['max total tasks'])):
                task_json_data = rnd.choice(env_setting_json['task settings'])
                auction_time = rnd.randint(0, total_time_steps)
                tasks.append(Task(
                    name='{} {}'.format(task_json_data['name'], task_num),
                    auction_time=auction_time,
                    deadline=auction_time + rnd.randint(task_json_data['min deadline'], task_json_data['max deadline']),
                    required_storage=float(rnd.randint(task_json_data['min required storage'],
                                                       task_json_data['max required storage'])),
                    required_comp=float(rnd.randint(task_json_data['min required computation'],
                                                    task_json_data['max required computation'])),
                    required_results_data=float(rnd.randint(task_json_data['min required results data'],
                                                            task_json_data['max required results data']))))

        return env_name, servers, tasks, total_time_steps
