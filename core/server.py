"""Implementation of a server with a fix amount of available resources at each time step"""

from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

import core.log as log

if TYPE_CHECKING:
    from agents.resource_weighting_agent import ResourceWeightingAgent
    from agents.task_pricing_agent import TaskPricingAgent
    from core.task import Task, TaskStage


class Server:
    """Server simulation class"""

    tasks: List[Task] = []

    storage_capacity: float = 0
    computational_capacity: float = 0
    bandwidth_capacity: float = 0

    task_pricing_agent: TaskPricingAgent = None
    resource_weighting_agent: ResourceWeightingAgent = None

    def __init__(self, name: str, storage_capacity: float, computational_capacity: float, bandwidth_capacity: float):
        self.name: str = name
        self.storage_capacity = storage_capacity
        self.computational_capacity = computational_capacity
        self.bandwidth_capacity = bandwidth_capacity

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def __str__(self):
        return f'{self.name} Server - Storage cap: {self.storage_capacity}, Comp cap: {self.computational_capacity}, ' \
               f'Bandwidth cap: {self.bandwidth_capacity}, TP agent: {self.task_pricing_agent.name}, ' \
               f"RW agent: {self.resource_weighting_agent.name}, tasks: {', '.join([task.name for task in self.tasks])}"

    def price_task(self, task: Task, time_step: int) -> float:
        assert self.task_pricing_agent is not None

        return self.task_pricing_agent.price_task(task, self.tasks, self, time_step)

    def allocate_task(self, task, second_min_price):
        self.tasks.append(task)

        self.task_pricing_agent.task_allocated(task, second_min_price)

    def allocate_resources(self, time_step: int, greedy: bool = True):
        log.debug(f'\tServer {self.name} resource weighting')

        if len(self.tasks) == 0:
            return
        elif len(self.tasks) == 1:
            task = self.tasks[0]
            if task.stage == TaskStage.LOADING:
                task.allocate_loading_resources(
                    min(self.bandwidth_capacity, task.required_storage - task.loading_progress, self.storage_capacity),
                    time_step)
            elif task.stage == TaskStage.COMPUTING:
                task.allocate_compute_resources(
                    min(self.computational_capacity, task.required_computation - task.compute_progress), time_step)
            elif task.stage == TaskStage.SENDING:
                task.allocate_sending_resources(
                    min(self.bandwidth_capacity, task.required_results_data - task.sending_results_progress), time_step)
            return

        loading_weights: Dict[Task, float] = {}
        compute_weights: Dict[Task, float] = {}
        sending_weights: Dict[Task, float] = {}

        # Stage 1: Finding the weighting for each of the tasks
        for task in self.tasks:
            weighting = self.resource_weighting_agent.weight_task(
                task, [_task for _task in self.tasks if task is not _task], self, time_step, greedy)
            log.debug(f'\t\tTask {task.name} {task.stage}: {weighting}')

            if task.stage == TaskStage.LOADING:
                loading_weights[task] = weighting
            elif task.stage == TaskStage.COMPUTING:
                compute_weights[task] = weighting
            elif task.stage == TaskStage.SENDING:
                sending_weights[task] = weighting

        available_storage: float = self.storage_capacity
        available_computation: float = self.computational_capacity
        available_bandwidth: float = self.bandwidth_capacity

        # Stage 2: Allocate the compute resources to tasks
        completed_compute_stage: bool = True
        while completed_compute_stage and compute_weights:
            compute_unit: float = available_computation / sum(compute_weights.values())
            completed_compute_stage = False

            for task, weight in compute_weights.items():
                if task.required_computation - task.compute_progress <= weight * compute_unit:
                    compute_resources: float = task.required_computation - task.compute_progress

                    task.allocate_compute_resources(compute_resources, time_step)
                    available_computation -= compute_resources
                    available_storage -= task.loading_progress

                    completed_compute_stage = True
                    compute_weights.pop(task)

        if compute_weights:
            compute_unit = available_computation / sum(compute_weights.values())
            for task, weight in compute_weights.items():
                task.allocate_compute_resources(compute_unit * weight, time_step)

        # Stage 3: Allocate the bandwidth resources to task
        completed_bandwidth_stage: bool = True
        while completed_bandwidth_stage and (loading_weights or sending_weights):
            bandwidth_unit: float = available_bandwidth / (
                    sum(loading_weights.values()) + sum(sending_weights.values()))
            completed_bandwidth_stage = False

            for task, weight in sending_weights.items():
                if task.required_results_data - task.sending_results_progress <= weight * bandwidth_unit:
                    sending_resources: float = task.required_results_data - task.sending_results_progress
                    task.allocate_sending_resources(sending_resources, time_step)

                    available_bandwidth -= sending_resources
                    available_storage -= task.loading_progress

                    completed_bandwidth_stage = True

                    sending_weights.pop(task)

            for task, weight in loading_weights.items():
                if task.required_storage - task.loading_progress <= weight * bandwidth_unit and \
                        task.loading_progress + min(task.required_storage - task.loading_progress,
                                                    weight * bandwidth_unit) <= available_storage:
                    loading_resources: float = task.required_storage - task.loading_progress
                    task.allocate_loading_resources(loading_resources, time_step)

                    available_bandwidth -= loading_resources
                    available_storage -= task.loading_progress

                    completed_bandwidth_stage = True

                    loading_weights.pop(task)

        if loading_weights or sending_weights:
            bandwidth_unit: float = available_bandwidth / (
                    sum(loading_weights.values()) + sum(sending_weights.values()))
            if loading_weights:
                for task, weight in loading_weights.items():
                    task.allocate_loading_resources(bandwidth_unit * weight, time_step)

            if sending_weights:
                for task, weight in sending_weights.items():
                    task.allocate_sending_resources(bandwidth_unit * weight, time_step)
