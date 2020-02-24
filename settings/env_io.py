"""Setting module for loading and saving environment and loading environment settings"""

from __future__ import annotations

import json
from random import randint, choice
from typing import List, TYPE_CHECKING, Tuple

from env.server import Server
from env.task import Task

if TYPE_CHECKING:
    from env.environment import OnlineFlexibleResourceAllocationEnv


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
        'env name': environment.env_name,
        'total time steps': environment.total_time_steps,
        'servers': [
            {'name': server.name, 'storage capacity': server.storage_cap, 'computational capacity': server.comp_cap,
             'bandwidth capacity': server.bandwidth_cap}
            for server in environment.state.server_tasks.keys()
        ],
        'tasks': [
            {'name': task.name, 'required storage': task.required_storage,
             'required computational': task.required_comp, 'required results data': task.required_results_data,
             'auction time': task.auction_time, 'deadline': task.deadline}
            for task in environment.unallocated_tasks
        ]
    }

    with open(filename, 'w') as file:
        json.dump(environment_json_data, file)


def load_setting(filename: str) -> Tuple[str, List[Server], List[Task], int]:
    """
    Load an environment settings from a file with a number of environments
    :param filename: The filename to loads the settings from
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

    with open(filename) as file:
        env_setting_json = json.load(file)

        env_name = env_setting_json['name']
        total_time_steps = randint(env_setting_json['min total time steps'],
                                   env_setting_json['max total time steps'])
        servers: List[Server] = []
        for server_num in range(randint(env_setting_json['min total servers'],
                                        env_setting_json['max total servers'])):
            server_json_data = choice(env_setting_json['server settings'])
            servers.append(Server(name='{} {}'.format(server_json_data['name'], server_num),
                                  storage_cap=randint(server_json_data['min storage capacity'],
                                                      server_json_data['max storage capacity']),
                                  comp_cap=randint(server_json_data['min computational capacity'],
                                                   server_json_data['max computational capacity']),
                                  bandwidth_cap=randint(server_json_data['min bandwidth capacity'],
                                                        server_json_data['max bandwidth capacity'])))

        tasks: List[Task] = []
        for task_num in range(randint(env_setting_json['min total tasks'],
                                      env_setting_json['max total tasks'])):
            task_json_data = choice(env_setting_json['task settings'])
            auction_time = randint(0, total_time_steps)
            tasks.append(Task(name='{} {}'.format(task_json_data['name'], task_num), auction_time=auction_time,
                              deadline=auction_time + randint(task_json_data['min deadline'],
                                                              task_json_data['max deadline']),
                              required_storage=randint(task_json_data['min required storage'],
                                                       task_json_data['max required storage']),
                              required_comp=randint(task_json_data['min required computation'],
                                                    task_json_data['max required computation']),
                              required_results_data=randint(task_json_data['min required results data'],
                                                            task_json_data['max required results data'])))

    return env_name, servers, tasks, total_time_steps
