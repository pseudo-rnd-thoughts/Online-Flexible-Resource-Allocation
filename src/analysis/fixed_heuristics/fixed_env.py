"""
Fixed Resource allocation env
"""

from collections import namedtuple
from math import exp
from typing import List

from docplex.cp.model import CpoModel, SOLVE_STATUS_OPTIMAL, SOLVE_STATUS_FEASIBLE

from env.env_state import EnvState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task import Task

# Fixed task namedtuple
FixedTask = namedtuple('FixedTask', ['name', 'auction_time', 'deadline',
                                     'required_storage', 'required_computation', 'required_results_data',
                                     'fixed_loading_speed', 'fixed_compute_speed', 'fixed_sending_speed'])


def convert_fixed_task(tasks: List[Task]):
    """
    Converts tasks to fixed tasks

    Args:
        tasks: List of tasks

    Returns: List of fixed tasks
    """
    fixed_tasks = []
    for task in tasks:
        model = CpoModel('FixedTask')

        loading_speed = model.integer_var(min=1, name='loading speed')
        compute_speed = model.integer_var(min=1, name='compute speed')
        sending_speed = model.integer_var(min=1, name='sending speed')

        model.add((task.required_storage / loading_speed) +
                  (task.required_computation / compute_speed) +
                  (task.required_results_data / sending_speed) <= (task.deadline - task.auction_time))

        model.minimize(exp(loading_speed) + exp(compute_speed) + exp(sending_speed))

        model_solution = model.solve(log_output=None, TimeLimit=3)

        if model_solution.get_solve_status() != SOLVE_STATUS_FEASIBLE and model_solution.get_solve_status() != SOLVE_STATUS_OPTIMAL:
            fixed_tasks.append(FixedTask(name=task.name, auction_time=task.auction_time, deadline=task.deadline,
                                         required_storage=task.required_storage, required_computation=task.required_computation,
                                         required_results_data=task.required_results_data,
                                         fixed_loading_speed=model_solution.get_value(loading_speed),
                                         fixed_compute_speed=model_solution.get_value(compute_speed),
                                         fixed_sending_speed=model_solution.get_value(sending_speed)))
        else:
            print(f'Error: {model_solution.get_solve_status()}')

    return fixed_tasks


def fixed_resource_allocation_model(env: OnlineFlexibleResourceAllocationEnv, state: EnvState):
    """
    Generate the fixed resource allocation model and then solve it

    Args:
        env: Online Flexible Resource Allocation Env
        state: Environment state

    Returns: Cplex model
    """
    tasks = env._unallocated_tasks
    if state.auction_task:
        tasks.append(state.auction_task)
    fixed_tasks = convert_fixed_task(tasks)
    servers = list(state.server_tasks.keys())

    model = CpoModel('FixedEnv')

    server_task_allocation = {
        (server, task): model.binary_var(name=f'{server.name} server - {task.name} task')
        for server in servers for task in fixed_tasks
    }

    for task in fixed_tasks:
        model.add(sum(server_task_allocation[(server, task)] for server in servers) <= 1)

    for server in servers:
        for time in range(env._total_time_steps):
            model.add(sum(min(task.required_storage, task.fixed_loading_speed * (time + 1 - task.auction_time))
                          for task in fixed_tasks if task.auction_time <= time <= task.deadline))
            model.add(sum(task.fixed_compute_speed * server_task_allocation[(server, task)]
                          for task in fixed_tasks if task.auction_time <= time <= task.deadline))
            model.add(sum((task.fixed_loading_speed + task.fixed_sending_speed) * server_task_allocation[(server, task)]
                          for task in fixed_tasks if task.auction_time <= time <= task.deadline))

    model.maximize(sum(server_task_allocation[(server, task)] for server in servers for task in tasks))

    model_solution = model.solve(log_output=None, TimeLimit=300)

    total_tasks_completed = sum(model_solution.get_value(server_task_allocation[(server, task)])
                                for server in servers for task in fixed_tasks)

    return total_tasks_completed
