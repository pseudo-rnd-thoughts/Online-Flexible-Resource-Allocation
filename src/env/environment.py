"""
Environment for online flexible resource allocation
"""

from __future__ import annotations

import json
import operator
import random as rnd
from copy import deepcopy
from math import inf
from typing import TYPE_CHECKING, Optional, Sequence

import gym

from env.env_state import EnvState
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from typing import List, Dict, Union, Tuple

    ACTION_TYPE = Dict[Server, Union[float, Dict[Task, float]]]
    REWARD_TYPE = Dict[Server, Union[float, List[Task]]]


class OnlineFlexibleResourceAllocationEnv(gym.Env):
    """
    The environment that manages the high level working for online flexible resource allocation
    This is a multi-rl_agents mixed cooperative and competitive situation that aims for the agents
        to maximise the social welfare of the system.
    """

    def __init__(self, env_settings: Optional[Union[str, List[str]]], env_name: str = '',
                 server_tasks: Optional[Dict[Server, List[Task]]] = None, tasks: Sequence[Task] = (),
                 time_step: int = -1, total_time_steps: int = -1):
        """
        Constructor of the environment that allows that environment to either with a environment setting or
            as a new environment that can't be reset
        Args:
            env_settings: List of environment setting files
            env_name: The current environment name
            server_tasks: Optional List of server tasks
            tasks: Optional List of tasks
            time_step: Optional environment time steps
            total_time_steps: Optional environment total time steps
        """
        if env_settings:
            self.env_settings = [env_settings] if type(env_settings) is str else env_settings

            self.env_name, self._total_time_steps, self._unallocated_tasks, self._state = '', -1, [], None
        else:
            self.env_settings = []

            self.env_name = env_name
            self._total_time_steps = total_time_steps
            self._unallocated_tasks: List[tasks] = list(tasks)
            assert all(tasks[pos].auction_time <= tasks[pos+1].auction_time for pos in range(len(tasks)-1))
            if self._unallocated_tasks:
                assert time_step <= self._unallocated_tasks[0].auction_time

                if self._unallocated_tasks[0].auction_time == time_step:
                    auction_task = self._unallocated_tasks.pop(0)
                else:
                    auction_task = None
            else:
                auction_task = None
            self._state = EnvState(server_tasks, auction_task, time_step)

    def __str__(self) -> str:
        if self._total_time_steps == -1:
            return 'Environment hasn\'t been generated'
        else:
            unallocated_task_str = '\n\t'.join([str(task) for task in self._unallocated_tasks])
            server_tasks_str = ', '.join([f'{server.name}: [{", ".join([task.name for task in tasks])}]'
                                          for server, tasks in self._state.server_tasks.items()])
            auction_task_str = str(self._state.auction_task) if self._state.auction_task else 'None'
            return f'Env State ({hex(id(self))}) at time step: {self._state.time_step}\n' \
                   f'\tAuction Task -> {auction_task_str}\n' \
                   f'\tServers -> {{{server_tasks_str}}}\n' \
                   f'\tUnallocated tasks: \n\t{unallocated_task_str}'

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def render(self, mode='human'):
        """
        Renders the environment to a graph

        Args:
            mode: The human is observation from
        """
        raise NotImplementedError('This has not been implemented yet')

    def reset(self) -> EnvState:
        """
        Resets the environment using one of the environment env_settings that is randomly chosen

        Returns: The new environment state

        """
        assert 0 < len(self.env_settings)

        # Select the env setting and load the environment env_settings
        env_setting: str = rnd.choice(self.env_settings)
        env_name, new_servers, new_tasks, new_total_time_steps = self._load_setting(env_setting)

        # Update the environment variables
        self.env_name = env_name
        self._total_time_steps = new_total_time_steps
        self._unallocated_tasks: List[Task] = sorted(new_tasks, key=operator.attrgetter('auction_time'))
        auction_task = self._unallocated_tasks.pop(0) if self._unallocated_tasks[0].auction_time == 0 else None
        self._state = EnvState({server: [] for server in new_servers}, auction_task, 0)

        return self._state

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
            assert all(server in actions for server in self._state.server_tasks.keys())
            assert all(type(action) is float for action in actions.values()), \
                ', '.join(str(type(action)) for action in actions.values())
            assert all(0 <= action for action in actions.values())

            # Vickrey auction, the server wins with the minimum price but only pays the second minimum price
            #  If multiple servers all price the same price then the server pays the minimum price (not second minimum price)
            min_price, min_servers, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price > 0:  # If the price is zero, then the bid is ignored
                    if price < min_price:
                        min_price, min_servers, second_min_price = price, [server], min_price
                    elif price == min_price:
                        min_servers.append(server)
                        second_min_price = price
                    elif price < second_min_price:
                        second_min_price = price

            # Creates the next environment state by copying the server task info, get the next auction task and the time step doesnt change
            next_state: EnvState = EnvState(deepcopy(self._state.server_tasks),
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
                    price = second_min_price if second_min_price < inf else min_price
                    rewards[winning_server] = price
                    updated_task = self._state.auction_task.assign_server(price, self._state.time_step)
                    next_state.server_tasks[winning_server].append(updated_task)
            else:
                info['min servers'] = 'failed, no server won'

        else:
            # Resource allocation (Action = Dict[Server, Dict[Task, float]])
            # Convert weights to resources
            info['step type'] = 'resource allocation'
            assert all(server in actions for server in self._state.server_tasks.keys())
            assert all(task in actions[server] for server, tasks in self._state.server_tasks.items() for task in tasks), \
                ', '.join([f'{server.name}: {task.name}' for server, tasks in self._state.server_tasks.items()
                           for task in tasks if task not in actions[server]])
            assert all(type(actions[server][task]) is float and 0 <= actions[server][task]
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
                    next_server_tasks[server], rewards[server] = [], []

            assert sum(len(tasks) for tasks in self._state.server_tasks.values()) == sum(
                len(tasks) for tasks in next_server_tasks.values()) + sum(len(tasks) for tasks in rewards.values())

            # The updated state
            next_state = EnvState(next_server_tasks,
                                  self._next_auction_task(self._state.time_step + 1),
                                  self._state.time_step + 1)

        # Check that all active task are within the valid time step
        assert all(task.auction_time <= next_state.time_step <= task.deadline
                   for server, tasks in next_state.server_tasks.items() for task in tasks), next_state
        # Painful to execute O(n^2) but just checks that all tasks that are modified
        assert all(id(task) != id(_task)
                   for tasks in self._state.server_tasks.values() for task in tasks
                   for _tasks in next_state.server_tasks.values() for _task in _tasks)
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING
                   for server, tasks in next_state.server_tasks.items() for task in tasks)
        for server, tasks in next_state.server_tasks.items():
            for task in tasks:
                task.assert_valid()

        self._state = next_state
        return self._state, rewards, self._total_time_steps < self._state.time_step, info

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

    def save_env(self, filename: str):
        """
        Saves this environment to a file with the template in settings/format.env

        Args:
            filename: The filename to save the environment to

        """
        # Check that the environment is valid
        for server, tasks in self._state.server_tasks.items():
            server.assert_valid()
            for task in tasks:
                task.assert_valid()
        for task in self._unallocated_tasks:
            task.assert_valid()

        # Add the auction task to the beginning of the unallocated task list
        tasks = ([] if self._state.auction_task is None else [self._state.auction_task]) + self._unallocated_tasks

        # Generate the environment JSON data
        env_json_data = {
            'env name': self.env_name,
            'time step': self._state.time_step,
            'total time steps': self._total_time_steps,
            'servers': [
                {
                    'name': server.name, 'storage capacity': server.storage_cap,
                    'computational capacity': server.computational_cap, 'bandwidth capacity': server.bandwidth_cap,
                    'tasks': [
                        {
                            'name': task.name, 'required storage': task.required_storage,
                            'required computational': task.required_computation,
                            'required results data': task.required_results_data, 'auction time': task.auction_time,
                            'deadline': task.deadline, 'stage': task.stage.name,
                            'loading progress': task.loading_progress, 'compute progress': task.compute_progress,
                            'sending progress': task.sending_progress, 'price': task.price
                        }
                        for task in tasks
                    ]
                }
                for server, tasks in self._state.server_tasks.items()
            ],
            'unallocated tasks': [
                {
                    'name': task.name, 'required storage': task.required_storage,
                    'required computational': task.required_computation,
                    'required results data': task.required_results_data, 'auction time': task.auction_time,
                    'deadline': task.deadline
                }
                for task in tasks
            ]
        }

        with open(filename, 'w') as file:
            json.dump(env_json_data, file)

    @staticmethod
    def load_env(filename: str):
        """
        Loads an environment from a file from template file at settings/format.env

        Args:
            filename: The filename to load the environment from

        Returns: The loaded environment

        """

        with open(filename) as file:
            json_data = json.load(file)

            name: str = json_data['env name']
            time_step: int = json_data['time step']
            total_time_steps: int = json_data['total time steps']

            # Load the servers list
            server_tasks: Dict[Server, List[Task]] = {
                Server(name=server_data['name'], storage_cap=server_data['storage capacity'],
                       computational_cap=server_data['computational capacity'],
                       bandwidth_cap=server_data['bandwidth capacity']): [
                    Task(name=task_data['name'], auction_time=task_data['auction time'], deadline=task_data['deadline'],
                         required_storage=task_data['required storage'],
                         required_computation=task_data['required computational'],
                         required_results_data=task_data['required results data'],
                         stage=TaskStage[task_data['stage']], loading_progress=task_data['loading progress'],
                         compute_progress=task_data['compute progress'], sending_progress=task_data['sending progress'],
                         price=task_data['price'])
                    for task_data in server_data['tasks']
                ]
                for server_data in json_data['servers']
            }

            for server, tasks in server_tasks.items():
                server.assert_valid()
                for task in tasks:
                    task.assert_valid()

            # Load the unallocated task list
            unallocated_tasks: List[Task] = [
                Task(name=task_data['name'], auction_time=task_data['auction time'], deadline=task_data['deadline'],
                     required_storage=task_data['required storage'],
                     required_computation=task_data['required computational'],
                     required_results_data=task_data['required results data'])
                for task_data in json_data['unallocated tasks']
            ]

        env = OnlineFlexibleResourceAllocationEnv(None, env_name=name, server_tasks=server_tasks,
                                                  tasks=unallocated_tasks, time_step=time_step,
                                                  total_time_steps=total_time_steps)
        return env, env._state

    @staticmethod
    def _load_setting(filename: str) -> Tuple[str, List[Server], List[Task], int]:
        """
        Load an environment env_settings from a file with a number of environments with the following template

        Args:
            filename: The filename to loads the env_settings from

        Returns: Returns the primary features of an environment to be set

        """

        with open(filename) as file:
            env_setting_json = json.load(file)

            env_name = env_setting_json['name']
            assert env_name != ''
            total_time_steps = rnd.randint(env_setting_json['min total time steps'],
                                           env_setting_json['max total time steps'])
            assert 0 < total_time_steps

            servers: List[Server] = []
            for server_num in range(rnd.randint(env_setting_json['min total servers'],
                                                env_setting_json['max total servers'])):
                server_json_data = rnd.choice(env_setting_json['server settings'])
                server = Server(
                    name='{} {}'.format(server_json_data['name'], server_num),
                    storage_cap=float(rnd.randint(server_json_data['min storage capacity'],
                                                  server_json_data['max storage capacity'])),
                    computational_cap=float(rnd.randint(server_json_data['min computational capacity'],
                                                        server_json_data['max computational capacity'])),
                    bandwidth_cap=float(rnd.randint(server_json_data['min bandwidth capacity'],
                                                    server_json_data['max bandwidth capacity'])))
                server.assert_valid()
                servers.append(server)

            tasks: List[Task] = []
            for task_num in range(rnd.randint(env_setting_json['min total tasks'],
                                              env_setting_json['max total tasks'])):
                task_json_data = rnd.choice(env_setting_json['task settings'])
                auction_time = rnd.randint(0, total_time_steps)
                task = Task(
                    name='{} {}'.format(task_json_data['name'], task_num),
                    auction_time=auction_time,
                    deadline=auction_time + rnd.randint(task_json_data['min deadline'], task_json_data['max deadline']),
                    required_storage=float(rnd.randint(task_json_data['min required storage'],
                                                       task_json_data['max required storage'])),
                    required_computation=float(rnd.randint(task_json_data['min required computation'],
                                                           task_json_data['max required computation'])),
                    required_results_data=float(rnd.randint(task_json_data['min required results data'],
                                                            task_json_data['max required results data'])))
                task.assert_valid()
                tasks.append(task)

        return env_name, servers, tasks, total_time_steps

    @staticmethod
    def custom_env(env_name: str, total_time_steps: int, new_servers_tasks: Dict[Server, List[Task]],
                   new_unallocated_tasks: List[Task]):
        """
        Setup a custom environment

        Args:
            env_name: New environment name
            total_time_steps: The total time steps of the environment
            new_servers_tasks: A dictionary of server to list of tasks
            new_unallocated_tasks: A list of unallocated tasks

        Returns: A tuple of new environment and its state

        """

        # Check that the inputs are valid
        assert 0 < total_time_steps
        assert 0 < len(new_servers_tasks)
        assert all(task.stage is not TaskStage.UNASSIGNED or task.stage is not TaskStage.COMPLETED
                   or task.stage is not TaskStage.FAILED for _, tasks in new_servers_tasks.items() for task in tasks)
        assert all(task.stage is TaskStage.UNASSIGNED for task in new_unallocated_tasks)
        for task in new_unallocated_tasks:
            task.assert_valid()
        for server, tasks in new_servers_tasks.items():
            server.assert_valid()
            for task in tasks:
                task.assert_valid()

        env = OnlineFlexibleResourceAllocationEnv(None, env_name=env_name, total_time_steps=total_time_steps,
                                                  server_tasks=new_servers_tasks, tasks=new_unallocated_tasks)

        return env, env._state
