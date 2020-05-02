"""
Tests the resource allocation in the server class for both bandwidth/storage allocation and compute allocation
"""

from __future__ import annotations

import random as rnd

from env.server import Server
from env.task import Task
from env.task_stage import TaskStage


def test_allocate_compute_resources(error_term=0.1):
    tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0),
        Task('Test 4', 69.0, 35.0, 10.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=69.0, compute_progress=20.0)
    ]
    server = Server('Test', 220.0, 35.0, 22.0)

    max_iterations, it = 10, 0
    while tasks and it < max_iterations:
        task_weights = {task: rnd.randint(1, 25) for task in tasks}

        task_resource_usage = server.allocate_compute_resources(task_weights, server.computational_cap, 0)

        assert sum(compute_usage for (_, compute_usage, _) in task_resource_usage.values()) < server.computational_cap + error_term

        print(f'\nTask resource usage')
        for task, (storage_usage, comp_usage, bandwidth_usage) in task_resource_usage.items():
            print(f'\t{task.name} Task - Comp Usage: {comp_usage}, Comp Progress: {task.compute_progress}, '
                  f'Required Comp: {task.required_computation} Stage: {task.stage}')
        print()

        tasks = [task for task in task_resource_usage.keys() if task.stage is TaskStage.COMPUTING]
        it += 1


def test_allocate_bandwidth_resources(error_term=0.1):
    tasks = [
        Task('Test 1', 56.0, 55.0, 15.0, 0, 9, stage=TaskStage.LOADING, loading_progress=15),
        Task('Test 2', 75.0, 39.0, 18.0, 0, 12, stage=TaskStage.LOADING, loading_progress=50),
        Task('Test 3', 52.0, 30.0, 26.0, 0, 9, stage=TaskStage.LOADING, loading_progress=25),
        Task('Test 4', 60.0, 52.0, 18.0, 0, 12, stage=TaskStage.SENDING, loading_progress=60.0, compute_progress=52.0),
        Task('Test 5', 57.0, 56.0, 15.0, 0, 10, stage=TaskStage.SENDING, loading_progress=57.0, compute_progress=56.0, sending_progress=10.0),
        Task('Test 6', 72.0, 32.0, 23.0, 0, 12, stage=TaskStage.SENDING, loading_progress=72.0, compute_progress=32.0, sending_progress=12.0)
    ]
    server = Server('Test', 310, 29, 40)

    max_iterations, it = 10, 0
    while tasks and it < max_iterations:
        loading_weights = {task: rnd.randint(1, 25) for task in tasks if task.stage is TaskStage.LOADING}
        sending_weights = {task: rnd.randint(1, 25) for task in tasks if task.stage is TaskStage.SENDING}

        available_storage = server.storage_cap - sum(task.loading_progress for task in loading_weights.keys()) - \
            sum(task.loading_progress for task in sending_weights.keys())

        task_resource_usage = server.allocate_bandwidth_resources(loading_weights, sending_weights, available_storage,
                                                                  server.bandwidth_cap, 0)

        assert sum(storage_usage for (storage_usage, _, _) in task_resource_usage.values()) < server.storage_cap + error_term
        assert sum(bandwidth_usage for (_, _, bandwidth_usage) in task_resource_usage.values()) < server.bandwidth_cap + error_term

        print(f'\nTask resource usage')
        for task, (storage_usage, comp_usage, bandwidth_usage) in task_resource_usage.items():
            if task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING:
                print(f'{task.name} Task - Bandwidth Usage: {bandwidth_usage}, Loading Progress: {task.loading_progress}, '
                      f'Required storage: {task.required_storage} Stage: {task.stage}')
            else:
                print(f'{task.name} Task - Bandwidth Usage: {bandwidth_usage}, Sending Progress: {task.sending_progress}, '
                      f'Required results data: {task.required_results_data} Stage: {task.stage}')

        tasks = [task for task in task_resource_usage.keys() if task.stage is TaskStage.LOADING or task.stage is TaskStage.SENDING]
        it += 1
