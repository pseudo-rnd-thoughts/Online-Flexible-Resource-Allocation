"""Immutable server class"""

from __future__ import annotations

from typing import NamedTuple, Dict, TYPE_CHECKING, Tuple

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task


class Server(NamedTuple):
    name: str

    storage_cap: int
    comp_cap: int
    bandwidth_cap: int

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.comp_cap}, Bandwidth cap: {self.bandwidth_cap}'

    def allocate_resources(self, resource_weights: Dict[Task, float]) -> Dict[Task, float]:
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

            available_storage: float = self.storage_cap
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

                        task_resource_usage[task.compute(compute_resources)] = (task.required_storage, compute_resources, 0)
                        available_computation -= compute_resources
                        available_storage -= task.required_storage

                        completed_compute_stage = True
                        compute_weights.pop(task)

            if compute_weights:
                compute_unit = available_computation / sum(compute_weights.values())
                for task, weight in compute_weights.items():
                    available_storage -= task.required_storage
                    task_resource_usage[task.compute(compute_unit * weight)] = (task.required_storage, compute_unit * weight, 0)

            # Stage 3: Allocate the bandwidth resources to task
            completed_bandwidth_stage: bool = True
            while completed_bandwidth_stage and (loading_weights or sending_weights):
                bandwidth_unit: float = available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values()))
                completed_bandwidth_stage = False

                for task, weight in sending_weights.items():
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        sending_resources: float = task.required_results_data - task.sending_progress

                        task_resource_usage[task.sending(sending_resources)] = (task.required_storage, 0, sending_resources)
                        available_bandwidth -= sending_resources
                        available_storage -= task.required_storage

                        completed_bandwidth_stage = True
                        sending_weights.pop(task)

                for task, weight in loading_weights.items():
                    if task.required_storage - task.loading_progress <= weight * bandwidth_unit and \
                            task.loading_progress + min(task.required_storage - task.loading_progress, weight * bandwidth_unit) <= available_storage:

                        loading_resources: float = task.required_storage - task.loading_progress
                        task.loading(loading_resources)
                        available_bandwidth -= loading_resources
                        available_storage -= task.loading_progress

                        completed_bandwidth_stage = True

                        loading_weights.pop(task)

            if loading_weights or sending_weights:
                bandwidth_unit: float = available_bandwidth / (
                        sum(loading_weights.values()) + sum(sending_weights.values()))
                if loading_weights:
                    for task, weight in loading_weights.items():
                        task.loading(bandwidth_unit * weight)

                if sending_weights:
                    for task, weight in sending_weights.items():

                        task.sending(bandwidth_unit * weight)
        else:
            # There are no task therefore nothing to allocate resources to
            return {}
