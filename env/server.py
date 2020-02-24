"""Immutable server class"""

from __future__ import annotations

from typing import NamedTuple, Dict, TYPE_CHECKING, Tuple, List

import core.log as log
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task


class Server(NamedTuple):
    """Server class with a name and resource capacity"""

    name: str

    storage_cap: int
    comp_cap: int
    bandwidth_cap: int

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.comp_cap}, Bandwidth cap: {self.bandwidth_cap}'

    def allocate_resources(self, resource_weights: Dict[Task, float], time_step: int) -> Tuple[List[Task], List[Task]]:
        if resource_weights:
            task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}  # The resource allocated to the task

            loading_weights: Dict[Task, float] = {}
            compute_weights: Dict[Task, float] = {}
            sending_weights: Dict[Task, float] = {}

            for task, weight in resource_weights.items():
                if task.stage is TaskStage.LOADING:
                    loading_weights[task] = weight
                elif task.stage is TaskStage.COMPUTING:
                    compute_weights[task] = weight
                elif task.stage is TaskStage.SENDING:
                    sending_weights[task] = weight

            available_storage: float = self.storage_cap - sum(task.loading_progress for task in resource_weights.keys())
            available_computation: float = self.comp_cap
            available_bandwidth: float = self.bandwidth_cap

            # Stage 2: Allocate the compute resources to tasks
            completed_compute_stage: bool = True
            while completed_compute_stage and compute_weights:
                compute_unit: float = available_computation / sum(compute_weights.values())
                completed_compute_stage = False

                for task, weight in compute_weights.items():
                    if task.required_comp - task.compute_progress <= weight * compute_unit:
                        compute_resources: float = task.required_storage - task.compute_progress

                        task_resource_usage[task.compute(compute_resources, time_step)] = (
                            task.required_storage, compute_resources, 0)
                        available_computation -= compute_resources

                        completed_compute_stage = True
                        compute_weights.pop(task)

            if compute_weights:
                compute_unit = available_computation / sum(compute_weights.values())
                for task, weight in compute_weights.items():
                    available_storage -= task.required_storage
                    task_resource_usage[task.compute(compute_unit * weight, time_step)] = (
                        task.required_storage, compute_unit * weight, 0)

            # Stage 3: Allocate the bandwidth resources to task
            completed_bandwidth_stage: bool = True
            while completed_bandwidth_stage and (loading_weights or sending_weights):
                bandwidth_unit: float = available_bandwidth / (
                        sum(loading_weights.values()) + sum(sending_weights.values()))
                completed_bandwidth_stage = False

                for task, weight in sending_weights.items():
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        sending_resources: float = task.required_results_data - task.sending_progress

                        task_resource_usage[task.sending(sending_resources, time_step)] = (
                            task.required_storage, 0, sending_resources)
                        available_bandwidth -= sending_resources

                        completed_bandwidth_stage = True
                        sending_weights.pop(task)

                for task, weight in loading_weights.items():
                    if task.required_storage - task.loading_progress <= weight * bandwidth_unit and \
                            task.loading_progress + min(task.required_storage - task.loading_progress,
                                                        weight * bandwidth_unit) <= available_storage:
                        loading_resources: float = task.required_storage - task.loading_progress
                        task_resource_usage[task.loading(loading_resources, time_step)] = (
                            task.required_storage, 0, loading_resources)
                        available_bandwidth -= loading_resources
                        available_storage -= loading_resources

                        completed_bandwidth_stage = True
                        loading_weights.pop(task)

            if loading_weights or sending_weights:
                bandwidth_unit: float = available_bandwidth / (
                        sum(loading_weights.values()) + sum(sending_weights.values()))
                for task, weight in loading_weights.items():
                    loading_resources = bandwidth_unit * weight
                    task_resource_usage[task.loading(loading_resources, time_step)] = (
                        task.loading_progress + loading_resources, 0, loading_resources)
                    available_storage -= loading_resources
                    available_bandwidth -= loading_resources

                for task, weight in sending_weights.items():
                    sending_resources = bandwidth_unit * weight
                    task_resource_usage[task.loading(sending_resources, time_step)] = (
                        task.required_storage, 0, sending_resources)
                    available_bandwidth -= sending_resources

            assert sum(resources_usage[0] for resources_usage in task_resource_usage.values()) <= self.storage_cap
            assert sum(resources_usage[1] for resources_usage in task_resource_usage.values()) <= self.comp_cap
            assert sum(resources_usage[2] for resources_usage in task_resource_usage.values()) <= self.bandwidth_cap

            log.debug(f'{self.name} Server resource usage -> ' + ', '.join(
                [f'{task.name} Task: ({storage_usage:.3f}, {compute_usage:.3f}, {bandwidth_usage:.3f})'
                 for task, (storage_usage, compute_usage, bandwidth_usage) in task_resource_usage.items()]))

            unfinished_tasks = [task for task in task_resource_usage.keys() if
                                task.stage is not TaskStage.COMPLETED or task.stage is not TaskStage.FAILED]
            completed_tasks = [task for task in task_resource_usage.keys() if
                               task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]

            return unfinished_tasks, completed_tasks
        else:
            # There are no task therefore nothing to allocate resources to
            return [], []
