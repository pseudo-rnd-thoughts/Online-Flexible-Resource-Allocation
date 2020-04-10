"""
Implementation of an online flexible resource allocation env in the style of an OpenAI Gym Environment
"""

import json
import operator
import random as rnd
from enum import Enum, auto
from math import inf
from typing import Optional, List, Dict, Tuple, Union, Sequence

import gym
import numpy as np

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


class StepType(Enum):
    """
    Online Flexible Resource allocation env step type is either an auction step or a resource allocation step
    """
    AUCTION = auto()
    RESOURCE_ALLOCATION = auto()


class OnlineFlexibleResourceAllocationEnv(gym.Env):
    """
    Gym environment for the online flexible resource allocation environment
    """

    def __init__(self, filenames: Union[str, List[str]], env_name: str = '', servers: Sequence[Server] = (),
                 server_tasks: Optional[Dict[Server, List[Task]]] = None, tasks: Sequence[Task] = (),
                 time_step: int = -1, total_time_steps: int = -1):
        self.environment_settings: List[str] = [filenames] if type(filenames) is str else filenames

        self.env_name: str = env_name

        self.time_step: int = time_step
        self.total_time_steps: int = total_time_steps

        self.unallocated_tasks: List[Task] = sorted(list(tasks), key=operator.attrgetter('auction_time'))
        self.auction_task: Optional[Task] = None
        self.server_tasks: Dict[Server, List[Task]] = {server: [] for server in servers} \
            if server_tasks is None else server_tasks

    def step(self, actions):
        """
        Steps over a single time step of the environment using the actions provided

        Args:
            actions: The actions for the environment, the shape of their actions is dependant

        Returns: The

        """
        assert self.time_step < self.total_time_steps
        assert type(actions) is dict
        assert all(server in actions for server in self.server_tasks.keys())

        rewards = {}
        if self.auction_task:
            actions: Dict[Server, float] = actions
            # Check the actions are valid
            assert all(type(action) is float for action in actions.values())
            assert all(0 <= action for action in actions.values())

            # Vickrey auction
            min_price, min_servers, second_min_price = inf, [], inf
            for server, price in actions.items():
                if price > 0:
                    if price < min_price:
                        min_price, min_servers, second_min_price = price, [server], min_price
                    elif price == min_price:
                        min_servers.append(server)

            if min_servers:
                winning_server: Server = rnd.choice(min_servers)
                price = second_min_price if len(min_servers) == 1 and second_min_price < inf else min_price
                self.auction_task.assign_server(price, self.time_step)
                self.server_tasks[winning_server].append(self.auction_task)
                rewards[winning_server] = price
        else:
            # Check the actions are valid
            actions: Dict[Server, List[float]] = actions
            assert all(type(server_actions) is list or type(server_actions) is np.ndarray
                       for server, server_actions in actions.items())
            assert all(type(task_action) is float or type(task_action) is np.float64
                       for server, server_actions in actions.items() for task_action in server_actions)
            assert all(0 <= task_action for server, server_actions in actions.items() for task_action in server_actions)

            # Allow each server to allocate resources to each task
            for server, resource_weights in actions.items():
                assert len(resource_weights) == len(self.server_tasks[server]), \
                    f'Resource length: {len(resource_weights)}, server tasks: {len(self.server_tasks[server])}'
                task_resource_weights: List[Tuple[Task, float]] = [
                    (task, resource_weight) for task, resource_weight in zip(self.server_tasks[server], resource_weights)
                ]
                self.server_tasks[server], rewards[server] = server.allocate_resources(task_resource_weights, self.time_step)

            self.time_step += 1

        return self._generate_state(), rewards, self.time_step > self.total_time_steps

    def reset(self):
        """
        Resets the environment using one of the environment env_settings that is randomly chosen

        Returns: The new environment state

        """
        assert 0 < len(self.environment_settings)

        # Select the env setting and load the environment env_settings
        env_setting: str = rnd.choice(self.environment_settings)
        env_name, new_servers, new_tasks, new_total_time_steps = self._load_setting(env_setting)

        assert 0 < len(new_tasks)
        assert 0 < len(new_servers)

        # Update the environment variables
        self.env_setting = env_setting
        self.env_name = env_name

        # Current state
        self.unallocated_tasks: List[Task] = sorted(new_tasks, key=operator.attrgetter('auction_time'))
        self.time_step = 0
        self.total_time_steps = new_total_time_steps
        self.server_tasks = {server: [] for server in new_servers}

        return self._generate_state()

    def render(self, mode='human'):
        """
        Renders the environment to a graph

        Args:
            mode: The human is observation from
        """
        raise NotImplementedError('This has not been implemented yet')

    def __str__(self):
        server_tasks_str = ', '.join([f'{server.name}: [{", ".join([task.name for task in tasks])}]'
                                      for server, tasks in self.server_tasks.items()])
        auction_task_str = str(self.auction_task) if self.auction_task else 'None'
        return f'Env State ({hex(id(self))}) at time step: {self.time_step}\n' \
               f'\tAuction Task -> {auction_task_str}\n' \
               f'\tServers -> {server_tasks_str}'

    def _generate_state(self) -> Tuple[Dict[Server, np.ndarray], StepType]:
        """
        Generates the state of the environment

        Returns: A numpy two dimensional array

        """
        self.auction_task = self._next_auction_task()

        if self.auction_task is not None:
            return {server: [self._normalise_task(self.auction_task, server) + [1.0]] +
                            [self._normalise_task(task, server) + [0.0] for task in tasks]
                    for server, tasks in self.server_tasks.items()}, StepType.AUCTION
        else:
            return {server: [self._normalise_task(task, server) for task in tasks]
                    for server, tasks in self.server_tasks.items()}, StepType.RESOURCE_ALLOCATION

    def _next_auction_task(self) -> Optional[Task]:
        """
        Generates the next auction task

        Returns: The next task

        """
        assert 0 <= self.time_step

        if self.unallocated_tasks:
            assert self.time_step <= self.unallocated_tasks[0].auction_time, \
                f'Top unallocated task auction time {self.unallocated_tasks[0].auction_time} at time step: {self.time_step}'
            return self.unallocated_tasks.pop(0) if self.unallocated_tasks[0].auction_time == self.time_step else None

    def _normalise_task(self, task: Task, server: Server) -> List[float]:
        """
        Normalises the task that is running on Server at environment time step

        Args:
            task: The task to be normalised
            server: The server that is the task is running on

        Returns: A list of floats where the task attributes are normalised

        """
        return [
            task.required_storage / server.storage_cap,
            task.required_storage / server.bandwidth_cap,
            task.required_computation / server.computational_comp,
            task.required_results_data / server.bandwidth_cap,
            task.deadline - self.time_step,
            task.loading_progress,
            task.compute_progress,
            task.sending_progress
        ]

    def save_env(self, filename: str):
        """
        Saves this environment to a file with the template in settings/format.env

        Args:
            filename: The filename to save the environment to

        """
        # Check that the environment is valid
        for server, tasks in self.server_tasks.items():
            server.assert_valid()
            for task in tasks:
                task.assert_valid()
        for task in self.unallocated_tasks:
            task.assert_valid()

        # Add the auction task to the beginning of the unallocated task list
        if self.auction_task is not None:
            self.unallocated_tasks.insert(0, self.auction_task)

        # Generate the environment JSON data
        env_json_data = {
            'env name': self.env_name,
            'total time steps': self.total_time_steps,
            'servers': [
                {
                    'name': server.name, 'storage capacity': server.storage_cap,
                    'computational capacity': server.computational_comp, 'bandwidth capacity': server.bandwidth_cap,
                    'tasks': [
                        {
                            'name': task.name, 'required storage': task.required_storage,
                            'required computational': task.required_computation,
                            'required results data': task.required_results_data, 'auction time': task.auction_time,
                            'deadline': task.deadline, 'stage': task.stage.name,
                            'loading progress': task.loading_progress, 'compute progress': task.compute_progress,
                            'sending progress': task.sending_progress, 'price': task.price
                        } for task in tasks
                    ]
                } for server, tasks in self.server_tasks.items()
            ],
            'unallocated tasks': [
                {
                    'name': task.name, 'required storage': task.required_storage,
                    'required computational': task.required_computation,
                    'required results data': task.required_results_data, 'auction time': task.auction_time,
                    'deadline': task.deadline
                } for task in self.unallocated_tasks
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
                       computational_comp=server_data['computational capacity'],
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
                for server_data in json_data['server']
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

        env = OnlineFlexibleResourceAllocationEnv(filename, env_name=name, server_tasks=server_tasks,
                                                  tasks=unallocated_tasks, time_step=time_step,
                                                  total_time_steps=total_time_steps)
        return env, env._generate_state()

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
                    required_computation=float(rnd.randint(task_json_data['min required computation'],
                                                           task_json_data['max required computation'])),
                    required_results_data=float(rnd.randint(task_json_data['min required results data'],
                                                            task_json_data['max required results data']))))

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

        env = OnlineFlexibleResourceAllocationEnv(filenames='', env_name=env_name, total_time_steps=total_time_steps,
                                                  server_tasks=new_servers_tasks, tasks=new_unallocated_tasks)

        return env, env._generate_state()
