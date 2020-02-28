"""Immutable server class"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import core.log as log
from core.core import assert_resource_allocation
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task
    from typing import Dict, Tuple, List


rounding = 0.0001


class Server(NamedTuple):
    """Server class with a name and resource capacity"""

    name: str

    storage_cap: float
    comp_cap: float
    bandwidth_cap: float

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.comp_cap}, Bandwidth cap: {self.bandwidth_cap}'

    @staticmethod
    def _round_down(value):
        return value - value % rounding

    def allocate_resources(self, resource_weights: Dict[Task, float], time_step: int) -> Tuple[List[Task], List[Task]]:
        if resource_weights:
            loading_weights: Dict[Task, float] = {}
            compute_weights: Dict[Task, float] = {}
            sending_weights: Dict[Task, float] = {}

            for task, weight in resource_weights.items():
                assert_resource_allocation(weight > 0)
                if task.stage is TaskStage.LOADING:
                    assert_resource_allocation(task.compute_progress == 0)
                    assert_resource_allocation(task.sending_progress == 0)
                    loading_weights[task] = weight
                elif task.stage is TaskStage.COMPUTING:
                    assert_resource_allocation(task.loading_progress >= task.required_storage)
                    assert_resource_allocation(task.sending_progress == 0)
                    compute_weights[task] = weight
                elif task.stage is TaskStage.SENDING:
                    assert_resource_allocation(task.loading_progress >= task.required_storage)
                    assert_resource_allocation(task.compute_progress >= task.required_comp)
                    sending_weights[task] = weight

            task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}
            available_storage = self.storage_cap - sum(task.loading_progress for task in resource_weights.keys())

            if loading_weights or sending_weights:
                bandwidth_unit = self.bandwidth_cap / (sum(weight for weight in loading_weights.values()) +
                                                       sum(weight for weight in sending_weights.values()))
                for task, sending_weight in sending_weights.items():
                    sending_resources = self._round_down(bandwidth_unit * sending_weight)
                    task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)
                for task, loading_weight in loading_weights.items():
                    loading_resources = min(self._round_down(bandwidth_unit * loading_weight), available_storage)
                    task_resource_usage[task.loading(loading_resources, time_step)] = (task.loading_progress + loading_resources, 0, loading_resources)
                    available_storage -= loading_resources

            if compute_weights:
                compute_unit = self.comp_cap / sum(weight for weight in compute_weights.values())
                for task, compute_weight in compute_weights.items():
                    compute_resources = self._round_down(compute_unit * compute_weight)
                    task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)

            self.log_task_resource_usage(resource_weights, task_resource_usage)
            self.assert_task_resource_usage(task_resource_usage)

            unfinished_tasks = [task for task in task_resource_usage.keys()
                                if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
            completed_tasks = [task for task in task_resource_usage.keys()
                               if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]
            return unfinished_tasks, completed_tasks
        else:
            return [], []

    def log_task_resource_usage(self, resource_weights: Dict[Task, float],
                                task_resource_usage: Dict[Task, Tuple[float, float, float]]):
        log.debug(f'{self.name} Server resource usage -> {{' + ', '.join(
            [f'{task.name} Task: ({storage_usage:.3f}, {compute_usage:.3f}, {bandwidth_usage:.3f})'
             for task, (storage_usage, compute_usage, bandwidth_usage) in task_resource_usage.items()]) + '}')
        log.debug(f'{self.name} Server task changes - {{', newline=False)
        for task in resource_weights.keys():
            modified_task = next(_task for _task in task_resource_usage.keys() if task == _task)
            if task.stage is TaskStage.LOADING:
                log.debug(
                    f'{task.name} Task: loading progress {task.loading_progress:.3f} -> {modified_task.loading_progress:.3f} '
                    f'({modified_task.stage}), ', newline=False)
            elif task.stage is TaskStage.COMPUTING:
                log.debug(
                    f'{task.name} Task: compute progress {task.compute_progress:.3f} -> {modified_task.compute_progress:.3f} '
                    f'({modified_task.stage}), ', newline=False)
            elif task.stage is TaskStage.SENDING:
                log.debug(
                    f'{task.name} Task: sending progress {task.sending_progress:.3f} -> {modified_task.sending_progress:.3f} '
                    f'({modified_task.stage}), ', newline=False)
        log.debug('}')

    def assert_task_resource_usage(self, task_resource_usage: Dict[Task, Tuple[float, float, float]]):
        for task, (storage_usage, compute_usage, sending_usage) in task_resource_usage.items():
            if task.stage is not TaskStage.FAILED:
                if task.stage is TaskStage.LOADING:
                    assert_resource_allocation(task.loading_progress < task.required_storage)
                    assert_resource_allocation(task.compute_progress == 0)
                    assert_resource_allocation(task.sending_progress == 0)
                else:
                    assert_resource_allocation(task.required_storage <= task.loading_progress)
                    assert_resource_allocation(task.required_storage <= storage_usage)
                    if task.stage is TaskStage.COMPUTING:
                        assert_resource_allocation(task.compute_progress < task.required_comp,
                                                   f'Failed {task.name} Task compute progress: {task.compute_progress} < required comp: {task.required_comp}, {str(task)}')
                        assert_resource_allocation(task.sending_progress == 0,
                                                   f'Failed {task.name} Task is Computing but has sending progress: {task.sending_progress}')
                    else:
                        assert_resource_allocation(task.required_comp <= task.compute_progress)
                        if task.stage is TaskStage.SENDING:
                            assert_resource_allocation(task.sending_progress < task.required_results_data)
                        else:
                            assert_resource_allocation(task.stage is TaskStage.COMPLETED)

        assert_resource_allocation(self._round_down(
            sum(resources_usage[0] for resources_usage in task_resource_usage.values())) <= self.storage_cap,
                                   f'{self.name} Server storage cap ({self.storage_cap}) failed as '
                                   f'{self._round_down(sum(resources_usage[0] for resources_usage in task_resource_usage.values()))}-> {{' +
                                   ', '.join([f'{task.name} Task: {storage_usage}' for task, (storage_usage, _, _) in
                                              task_resource_usage.items()]) + '}')
        assert_resource_allocation(self._round_down(
            sum(resources_usage[1] for resources_usage in task_resource_usage.values())) <= self.comp_cap,
                                   f'{self.name} Server computational cap ({self.comp_cap}) failed as '
                                   f'{self._round_down(sum(resources_usage[1] for resources_usage in task_resource_usage.values()))} -> {{' +
                                   ', '.join([f'{task.name} Task: {compute_usage}' for task, (_, compute_usage, _) in
                                              task_resource_usage.items()]) + '}')
        assert_resource_allocation(self._round_down(
            sum(resources_usage[2] for resources_usage in task_resource_usage.values())) <= self.bandwidth_cap,
                                   f'{self.name} Server bandwidth cap ({self.bandwidth_cap}) failed as '
                                   f'{self._round_down(sum(resources_usage[2] for resources_usage in task_resource_usage.values()))} -> {{' +
                                   ', '.join(
                                       [f'{task.name} Task: {bandwidth_usage}' for task, (_, _, bandwidth_usage) in
                                        task_resource_usage.items()]) + '}')

        assert_resource_allocation(all(
            task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING or
            task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED for task in task_resource_usage.keys()))

    def proper_allocate_resources(self, resource_weights: Dict[Task, float], time_step: int) -> Tuple[List[Task], List[Task]]:
        assert_resource_allocation(all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING for task in resource_weights.keys()),
                f'Failed {self.name} server task stage assert - Tasks: [' + ', '.join([f'{task.name} Task ({task.stage})' for task in resource_weights.keys()]) + ']')
        assert_resource_allocation(all(weight > 0 for weight in resource_weights.values()),
                f'Failed {self.name} server resource weight assert - Tasks: [' + ', '.join([f'{task.name} Task ({weight})' for task, weight in resource_weights.items()]) + ']')

        for task in resource_weights.keys():
            if task.stage is not TaskStage.FAILED:
                if task.stage is TaskStage.LOADING:
                    assert_resource_allocation(task.loading_progress < task.required_storage)
                    assert_resource_allocation(task.compute_progress == 0)
                    assert_resource_allocation(task.sending_progress == 0)
                else:
                    assert_resource_allocation(task.required_storage <= task.loading_progress)
                    if task.stage is TaskStage.COMPUTING:
                        assert_resource_allocation(task.compute_progress < task.required_comp,
                                f'Failed {task.name} Task compute progress: {task.compute_progress} < required comp: {task.required_comp}')
                        assert_resource_allocation(task.sending_progress == 0,
                                f'Failed {task.name} Task is Computing but has sending progress: {task.sending_progress}')
                    else:
                        assert_resource_allocation(task.required_comp <= task.compute_progress)
                        if task.stage is TaskStage.SENDING:
                            assert_resource_allocation(task.sending_progress < task.required_results_data)
                        else:
                            assert_resource_allocation(task.stage is TaskStage.COMPLETED)

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
                task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)
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
                compute_unit: float = available_computation / sum(compute_weights.values())
                completed_compute_stage = False

                for task, weight in compute_weights.items():
                    if task.required_comp - task.compute_progress <= weight * compute_unit:
                        compute_resources: float = task.required_comp - task.compute_progress

                        task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)
                        available_computation -= compute_resources

                        completed_compute_stage = True

                compute_weights = {task: weight for task, weight in compute_weights.items()
                                   if task not in list(task_resource_usage.keys())}

            if compute_weights:
                compute_unit = self._round_down(available_computation / sum(compute_weights.values()))
                for task, weight in compute_weights.items():
                    compute_resources = self._round_down(compute_unit * weight)

                    updated_task = task.compute(compute_resources, time_step)
                    task_resource_usage[updated_task] = (updated_task.required_storage, compute_resources, 0)
                    available_computation -= compute_resources

            # Stage 3: Allocate the bandwidth resources to task for both loading and sending data
            completed_bandwidth_stage: bool = True
            while completed_bandwidth_stage and (loading_weights or sending_weights):
                bandwidth_unit: float = self._round_down(available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())))
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
                            task.required_storage - task.loading_progress <= available_storage and \
                            task.required_storage - task.loading_progress <= available_bandwidth:
                        loading_resources: float = task.required_storage - task.loading_progress

                        task_resource_usage[task.loading(loading_resources, time_step)] = (task.required_storage, 0, loading_resources)

                        available_storage -= loading_resources
                        available_bandwidth -= loading_resources

                        completed_bandwidth_stage = True

                loading_weights = {task: weight for task, weight in loading_weights.items()
                                   if task not in list(task_resource_usage.keys())}

            if loading_weights or sending_weights:
                bandwidth_unit: float = self._round_down(available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())))
                for task, weight in loading_weights.items():
                    loading_resources = self._round_down(bandwidth_unit * weight)

                    task_resource_usage[task.loading(loading_resources, time_step)] = (task.loading_progress + loading_resources, 0, loading_resources)

                    available_storage -= loading_resources
                    available_bandwidth -= loading_resources

                for task, weight in sending_weights.items():
                    sending_resources = self._round_down(bandwidth_unit * weight)

                    task_resource_usage[task.loading(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)

                    available_bandwidth -= sending_resources

        self.log_task_resource_usage(resource_weights, task_resource_usage)
        self.assert_task_resource_usage(task_resource_usage)

        unfinished_tasks = [task for task in task_resource_usage.keys()
                            if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
        completed_tasks = [task for task in task_resource_usage.keys()
                           if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]
        return unfinished_tasks, completed_tasks
