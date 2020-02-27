"""Immutable server class"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import core.log as log
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task
    from typing import Dict, Tuple, List


rounding = 4


class Server(NamedTuple):
    """Server class with a name and resource capacity"""

    name: str

    storage_cap: int
    comp_cap: int
    bandwidth_cap: int

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.comp_cap}, Bandwidth cap: {self.bandwidth_cap}'

    def allocate_resources(self, resource_weights: Dict[Task, float], time_step: int) -> Tuple[List[Task], List[Task]]:
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is
                   TaskStage.SENDING for task in resource_weights.keys()), \
            f'Failed {self.name} server task stage assert - Tasks: [' + \
            ', '.join([f'{task.name} Task ({task.stage})' for task in resource_weights.keys()]) + ']'
        assert all(weight > 0 for weight in resource_weights.values()), \
            f'Failed {self.name} server resource weight assert - Tasks: [' + \
            ', '.join([f'{task.name} Task ({weight})' for task, weight in resource_weights.items()]) + ']'

        task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}  # The resource allocated to the task

        if len(resource_weights) == 0:
            log.debug(f'{self.name} Server - There is no resource weights provided')
            return [], []
        elif len(resource_weights) == 1:
            log.debug(f'{self.name} Server - There is a single task')

            task = next(_task for _task in resource_weights.keys())
            if task.stage is TaskStage.LOADING:
                loading_resources = min(self.storage_cap - task.loading_progress, self.bandwidth_cap, task.required_storage - task.loading_progress)
                task_resource_usage[task.loading(loading_resources, time_step)] = (task.loading_progress + loading_resources, 0, loading_resources)
            elif task.stage is TaskStage.COMPUTING:
                compute_resources = min(self.comp_cap - task.compute_progress, task.required_comp - task.compute_progress)
                task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)
            elif task.stage is TaskStage.SENDING:
                sending_resources = min(self.bandwidth_cap - task.sending_progress, task.required_results_data - task.sending_progress)
                task_resource_usage[task.loading(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)
        else:
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
                compute_unit: float = round(available_computation / sum(compute_weights.values()), rounding)
                completed_compute_stage = False

                for task, weight in compute_weights.items():
                    if task.required_comp - task.compute_progress <= weight * compute_unit:
                        compute_resources: float = task.required_storage - task.compute_progress

                        task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)
                        available_computation -= compute_resources

                        completed_compute_stage = True

                compute_weights = {task: weight for task, weight in compute_weights.items()
                                   if task not in list(task_resource_usage.keys())}

            if compute_weights:
                compute_unit = round(available_computation / sum(compute_weights.values()), rounding)
                for task, weight in compute_weights.items():
                    compute_resources = round(compute_unit * weight, rounding)

                    task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)
                    available_storage -= task.required_storage

            # Stage 3: Allocate the bandwidth resources to task
            completed_bandwidth_stage: bool = True
            while completed_bandwidth_stage and (loading_weights or sending_weights):
                bandwidth_unit: float = round(available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())), rounding)
                completed_bandwidth_stage = False

                for task, weight in sending_weights.items():
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        sending_resources: float = task.required_results_data - task.sending_progress

                        task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)
                        available_bandwidth -= sending_resources

                        completed_bandwidth_stage = True

                sending_weights = {task: weight for task, weight in sending_weights.items()
                                   if task not in list(task_resource_usage.keys())}

                for task, weight in loading_weights.items():
                    if task.required_storage - task.loading_progress <= weight * bandwidth_unit and \
                            task.loading_progress + min(task.required_storage - task.loading_progress, weight * bandwidth_unit) <= available_storage:
                        loading_resources: float = task.required_storage - task.loading_progress

                        task_resource_usage[task.loading(loading_resources, time_step)] = (task.required_storage, 0, loading_resources)

                        available_bandwidth -= loading_resources
                        available_storage -= loading_resources

                        completed_bandwidth_stage = True

                loading_weights = {task: weight for task, weight in loading_weights.items()
                                   if task not in list(task_resource_usage.keys())}

            if loading_weights or sending_weights:
                bandwidth_unit: float = round(available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())), rounding)
                for task, weight in loading_weights.items():
                    loading_resources = round(bandwidth_unit * weight, rounding)

                    task_resource_usage[task.loading(loading_resources, time_step)] = (task.loading_progress + loading_resources, 0, loading_resources)

                    available_storage -= loading_resources
                    available_bandwidth -= loading_resources

                for task, weight in sending_weights.items():
                    sending_resources = round(bandwidth_unit * weight, rounding)

                    task_resource_usage[task.loading(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)

                    available_bandwidth -= sending_resources

        assert sum(resources_usage[0] for resources_usage in task_resource_usage.values()) <= self.storage_cap, \
            f'{self.name} Server storage cap ({self.storage_cap}) failed -> {{' + \
            ', '.join([f'{task.name} Task: {storage_usage}' for task, (storage_usage, _, _) in task_resource_usage.items()]) + '}'
        assert sum(resources_usage[1] for resources_usage in task_resource_usage.values()) <= self.comp_cap, \
            f'{self.name} Server computational cap ({self.comp_cap}) failed -> {{' + \
            ', '.join([f'{task.name} Task: {compute_usage}' for task, (_, compute_usage, _) in task_resource_usage.items()]) + '}'
        assert sum(resources_usage[2] for resources_usage in task_resource_usage.values()) <= self.bandwidth_cap, \
            f'{self.name} Server bandwidth cap ({self.bandwidth_cap}) failed -> {{' + \
            ', '.join([f'{task.name} Task: {bandwidth_usage}' for task, (_, _, bandwidth_usage) in task_resource_usage.items()]) + '}'

        log.debug(f'{self.name} Server resource usage -> {{' + ', '.join(
            [f'{task.name} Task: ({storage_usage:.3f}, {compute_usage:.3f}, {bandwidth_usage:.3f})'
             for task, (storage_usage, compute_usage, bandwidth_usage) in task_resource_usage.items()]) + '}')
        log.debug(f'{self.name} Server task changes - {{', newline=False)
        for task in resource_weights.keys():
            modified_task = next(_task for _task in task_resource_usage.keys() if task == _task)
            if task.stage is TaskStage.LOADING:
                log.debug(
                    f'{task.name} Task: loading progress {task.loading_progress:.3f} -> {modified_task.loading_progress:.3f} '
                    f'({modified_task.stage}),', newline=False)
            elif task.stage is TaskStage.COMPUTING:
                log.debug(
                    f'{task.name} Task: compute progress {task.compute_progress:.3f} -> {modified_task.compute_progress:.3f} '
                    f'({modified_task.stage}),', newline=False)
            elif task.stage is TaskStage.SENDING:
                log.debug(
                    f'{task.name} Task: sending progress {task.sending_progress:.3f} -> {modified_task.sending_progress:.3f} '
                    f'({modified_task.stage}),', newline=False)
        log.debug('}')

        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING or
                   task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED for task in task_resource_usage.keys())

        unfinished_tasks = [task for task in task_resource_usage.keys()
                            if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
        completed_tasks = [task for task in task_resource_usage.keys()
                           if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]
        return unfinished_tasks, completed_tasks
