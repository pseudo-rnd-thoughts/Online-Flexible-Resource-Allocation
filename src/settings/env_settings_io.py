"""
Loading of environment settings to generate a new Online Flexible Resource Allocation Env
"""

import json
import operator
from typing import TYPE_CHECKING

from env.env_state import EnvState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.server import Server
from env.task import Task

if TYPE_CHECKING:
    from typing import List


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
    env.env_name = name
    env.total_time_steps = total_time_steps
    env.unallocated_tasks = sorted(tasks, key=operator.attrgetter('auction_time'))
    env.state = EnvState({server: [] for server in servers}, env.next_auction_task(0), 0)
    return env
