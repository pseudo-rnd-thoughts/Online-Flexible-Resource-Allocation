"""Setting module for loading and saving environment and loading environment settings"""

from __future__ import annotations

import json
from random import randint, choice
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.environment import OnlineFlexibleResourceAllocationEnv


def load_environment(filename: str) -> OnlineFlexibleResourceAllocationEnv:
    """
    Loads an environment from a file
    :param filename: The filename to load the environment from
    :return: The loaded environment

                Template
    {"name": "",
     "total time steps": 0,
     "servers": [{"name": "", "storage capacity": 0, "computational capacity": 0, "bandwidth capacity": 0}, ...],
     "tasks": [{"name": "", "required storage": 0, "required computation": 0, "required results data": 0,
                "auction time": 0, "deadline": 0}, ...]}
    """

    with open(filename) as file:
        json_data = json.load(file)

        total_time_steps: int = json_data['total time steps']
        servers: List[Task] = [
            (server_data['name'], server_data['storage capacity'], server_data['computational capacity'],
                   server_data['bandwidth capacity'])
            for server_data in json_data['servers']
        ]
        tasks: List[Task] = [
            (task_data['name'], task_data['auction time'], task_data['deadline'],
                 task_data['required storage'], task_data['required computational'], task_data['required results data'])
            for task_data in json_data['tasks']
        ]

    return servers, tasks, total_time_steps


def save_environment(environment: OnlineFlexibleResourceAllocationEnv, filename: str):
    """
    Saves an environment to a file
    :param environment: The environment to save
    :param filename: The filename to save the environment to

                Template
    {"name": "",
     "total time steps": 0,
     "servers": [{"name": "", "storage capacity": 0, "computational capacity": 0, "bandwidth capacity": 0}, ...],
     "tasks": [{"name": "", "required storage": 0, "required computation": 0, "required results data": 0,
                "auction time": 0, "deadline": 0}, ...]}
    """
    assert environment.time_step == 0

    environment_json_data = {
        'name': environment.current_env_setting,
        'total time steps': environment.total_time_steps,
        'servers': [
            {'name': server.name, 'storage capacity': server.storage_capacity,
             'computational capacity': server.computational_capacity, 'bandwidth capacity': server.bandwidth_capacity}
            for server in environment.state.keys()
        ],
        'tasks': [
            {'name': task.name, 'required storage': task.required_storage,
             'required computational': task.required_computation, 'required results data': task.required_results_data,
             'auction time': task.auction_time, 'deadline': task.deadline}
            for task in environment.unallocated_tasks
        ]
    }

    with open(filename, 'w') as file:
        json.dump(environment_json_data, file)


def load_environment_settings(filename: str, num_envs: int) -> List[OnlineFlexibleResourceAllocationEnv]:
    """
    Load an environment settings from a file with a number of environments
    :param filename: The filename to loads the settings from
    :param num_envs: The number of environment to create
    :return: A list of environments created using the environments

        Template
    {
      "name": "",
      "min total time steps": 0, "max total time steps": 0,
      "min total servers": 0, "max total servers": 0,
      "server settings": [
        {
          "name": "", "min storage capacity": 0, "max storage capacity": 0,
          "min computational capacity": 0, "max computational capacity": 0,
          "min bandwidth capacity": 0, "max bandwidth capacity": 0
        }, ...
      ],
      "task settings": [
        {
          "name": "", "min deadline": 0, "max deadline": 0,
          "min required storage": 0, "max required storage": 0,
          "min required computation": 0, "max required computation": 0,
          "min required results data": 0, "max required results data": 0
        }, ...
      ]
    }
    """

    environments: List[OnlineFlexibleResourceAllocationEnv] = []
    with open(filename) as file:
        env_setting_json = json.load(file)

    for env_num in range(num_envs):
        env_name = '{} {}'.format(env_setting_json['name'], env_num)
        total_time_steps = randint(env_setting_json['min total time steps'],
                                   env_setting_json['max total time steps'])
        servers: List[Server] = []
        for server_num in range(randint(env_setting_json['min total servers'],
                                        env_setting_json['max total servers'])):
            server_json_data = choice(env_setting_json['server settings'])
            servers.append(Server('{} {}'.format(server_json_data['name'], server_num),
                                  randint(server_json_data['min storage capacity'],
                                          server_json_data['max storage capacity']),
                                  randint(server_json_data['min computational capacity'],
                                          server_json_data['max computational capacity']),
                                  randint(server_json_data['min bandwidth capacity'],
                                          server_json_data['max bandwidth capacity'])))

        tasks: List[Task] = []
        for task_num in range(randint(env_setting_json['min total tasks'],
                                      env_setting_json['max total tasks'])):
            task_json_data = choice(env_setting_json['task settings'])
            auction_time = randint(0, total_time_steps)
            tasks.append(Task('{} {}'.format(task_json_data['name'], task_num), auction_time,
                              auction_time + randint(task_json_data['min deadline'], task_json_data['max deadline']),
                              randint(task_json_data['min required storage'],
                                      task_json_data['max required storage']),
                              randint(task_json_data['min required computation'],
                                      task_json_data['max required computation']),
                              randint(task_json_data['min required results data'],
                                      task_json_data['max required results data'])))

        environments.append(Environment(env_name, servers, tasks, total_time_steps))

    return environments
