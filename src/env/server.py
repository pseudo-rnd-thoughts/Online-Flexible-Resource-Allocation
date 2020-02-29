"""Immutable server class"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import core.log as log
from core.core import round_float
from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task
    from typing import Dict, Tuple, List


class Server(NamedTuple):
    """Server class with a name and resource capacity"""

    name: str

    storage_cap: float
    computational_comp: float
    bandwidth_cap: float

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.computational_comp}, ' \
               f'Bandwidth cap: {self.bandwidth_cap}'

    def allocate_resources(self, resource_weights: Dict[Task, float], time_step: int, error_term: float = 0.1) -> Tuple[List[Task], List[Task]]:
        """
        Allocate resources to tasks by converting a weighting (importance) to an actual resource
        :param resource_weights: A dictionary of task to weighting
        :param time_step: The current time step
        :param error_term: The error term to account for rounding effectively
        :return: Two list, the first being the list of completed or failed task, the second being tasks that are still ongoing
        """

        assert len(resource_weights) > 0
        # Assert that all tasks are at the correct stage
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING for task in resource_weights.keys())
        # Assert that all weights are greater than zero
        assert all(weight > 0 for weight in resource_weights.values())
        # Group the tasks by stage
        loading_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items() if task.stage is TaskStage.LOADING}
        compute_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items() if task.stage is TaskStage.COMPUTING}
        sending_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items() if task.stage is TaskStage.SENDING}

        # Resource available, storage is special because some storage is already used due to the previous stage
        available_storage: float = self.storage_cap - sum(task.loading_progress for task in resource_weights.keys())
        available_computation: float = self.computational_comp
        available_bandwidth: float = self.bandwidth_cap

        # Allocate computational resources to the tasks at computing stage
        compute_task_resource_usage = self.allocate_compute_resources(compute_weights, available_computation, time_step)
        # Allocate bandwidth resources (and storage) to the tasks at loading and sending stage
        bandwidth_task_resource_usage = self.allocate_bandwidth_resources(loading_weights, sending_weights, available_storage, available_bandwidth, time_step)

        # Join the compute and bandwidth resource allocation
        task_resource_usage = {**compute_task_resource_usage, **bandwidth_task_resource_usage}

        self.log_task_resource_usage(resource_weights, task_resource_usage)

        # Assert that the updated task are still valid
        for task in task_resource_usage.keys():
            task.assert_valid()
        # Assert that the resources used are less than available resources
        assert sum(storage_usage for (storage_usage, _, _) in task_resource_usage.values()) <= self.storage_cap + error_term
        assert sum(compute_usage for (_, compute_usage, _) in task_resource_usage.values()) <= self.computational_comp + error_term
        assert sum(bandwidth_usage for (_, _, bandwidth_usage) in task_resource_usage.values()) <= self.bandwidth_cap + error_term

        # Group the updated tasks in those completed or failed and those still ongoing
        unfinished_tasks = [task for task in task_resource_usage.keys() if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
        completed_tasks = [task for task in task_resource_usage.keys() if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]

        return unfinished_tasks, completed_tasks

    @staticmethod
    def allocate_compute_resources(compute_weights: Dict[Task, float], available_computation: float, time_step: int) -> Dict[Task, Tuple[float, float, float]]:
        """
        Allocate computational resources to tasks
        :param compute_weights: A dictionary of tasks (at computing stage) to weightings
        :param available_computation: The total available computation (= server.computational_cap)
        :param time_step: The current time step
        :return: A dictionary of tasks to their resource usage (storage, compute, bandwidth)
        """

        task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}

        assert all(task.stage is TaskStage.COMPUTING for task in compute_weights.keys())
        assert all(weight > 0 for weight in compute_weights.values())

        # Todo update such that this makes more sense
        # It is possible that the percentage of computational resources that could be allocated to a task is greater
        #   than the amount of required computational resource that the task needs to the allocated.
        # So this while loop, checks if the relative maximum computational resources that could be allocated to a
        #   task is less than the required computational resources
        # But in doing this, the relative maximum computational resources that are then available to other tasks has changed,
        #   so this process is looped over till no task will be completed with the relative maximum computational resources.
        task_updated: bool = True
        while task_updated and compute_weights:
            # Base unit of computational resources relative to the sum of weights for the compute resources
            compute_unit = round_float(available_computation / sum(compute_weights.values()))
            task_updated = False

            for task, weight in compute_weights.items():
                # If the weight compute resources are less than the needed computational resources, allocate only the required resources
                if task.required_comp - task.compute_progress <= weight * compute_unit:
                    compute_resources = round_float(task.required_comp - task.compute_progress)  # The required resources

                    # Set the updated task with the new resources and the resource used by the task
                    task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)
                    available_computation = round_float(available_computation - compute_resources)
                    task_updated = True

            # Update the compute weight based on the task that have had resources allocated (must be done here as dictionaries cant be used during the loop)
            compute_weights = {task: weight for task, weight in compute_weights.items() if task not in list(task_resource_usage.keys())}

        # If there are any tasks that their compute stage isn't completed using the compute unit
        if compute_weights:
            # The compute unit with the available computational resources leftover
            compute_unit = round_float(available_computation / sum(compute_weights.values()))

            for task, weight in compute_weights.items():
                # Updated the task with the compute resources
                compute_resources = round_float(compute_unit * weight)
                task_resource_usage[task.compute(compute_resources, time_step)] = (task.required_storage, compute_resources, 0)

        return task_resource_usage

    @staticmethod
    def allocate_bandwidth_resources(loading_weights: Dict[Task, float], sending_weights: Dict[Task, float],
                                     available_storage: float, available_bandwidth: float,
                                     time_step: int) -> Dict[Task, Tuple[float, float, float]]:
        """
        Allocate bandwidth (and storage) resources to task at Loading or Sending stages
        :param loading_weights: A dictionary of task (at loading stage) to weights
        :param sending_weights: A dictionary of task (at sending stage) to weights
        :param available_storage: The available storage of the server
        :param available_bandwidth: The available bandwidth of the server
        :param time_step: The current time step
        :return: A dictionary of tasks to resources used
        """

        task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}

        assert all(task.stage is TaskStage.LOADING for task in loading_weights.keys())
        assert all(weight > 0 for weight in loading_weights.values())
        assert all(task.stage is TaskStage.SENDING for task in sending_weights.keys())
        assert all(weight > 0 for weight in sending_weights.values())

        # Todo explain idea
        update_task: bool = True
        while update_task and (loading_weights or sending_weights):
            bandwidth_unit: float = round_float(available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())))
            update_task = False

            for task, weight in sending_weights.items():
                if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                    # Calculate the sending resources, update the task and resource usage, and bandwidth availability
                    sending_resources = round_float(task.required_results_data - task.sending_progress)
                    task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)
                    available_bandwidth = round_float(available_bandwidth - sending_resources)

                    update_task = True

            # Update the sending weights for tasks that havent had resources allocated (this cant happen during the loop with dictionaries)
            sending_weights = {task: weight for task, weight in sending_weights.items() if task not in list(task_resource_usage.keys())}

            # The repeat the same process for tasks being loaded
            for task, weight in loading_weights.items():
                # Check that the resources required to complete the loading stage is less than min available resources
                if task.required_storage - task.loading_progress <= min(weight * bandwidth_unit, available_storage, available_bandwidth):
                    loading_resources = round_float(task.required_storage - task.loading_progress)

                    task_resource_usage[task.loading(loading_resources, time_step)] = (task.required_storage, 0, loading_resources)
                    available_storage = round_float(available_storage - loading_resources)
                    available_bandwidth = round_float(available_bandwidth - loading_resources)

                    update_task = True

            loading_weights = {task: weight for task, weight in loading_weights.items() if task not in list(task_resource_usage.keys())}

        # If there are any tasks left then allocate the remaining tasks
        if loading_weights or sending_weights:
            bandwidth_total_weights = sum(loading_weights.values()) + sum(sending_weights.values())

            # Try to allocate resources for loading te
            for task, weight in loading_weights.items():
                # Calculate the loading resources available to the task
                loading_resources = round_float(min(round_float(available_bandwidth / bandwidth_total_weights * weight), available_storage))

                updated_task = task.loading(loading_resources, time_step)
                task_resource_usage[updated_task] = (updated_task.loading_progress, 0, loading_resources)

                available_storage = round_float(available_storage - loading_resources)
                available_bandwidth = round_float(available_bandwidth - loading_resources)
                bandwidth_total_weights -= weight

            update_task = True
            while update_task and sending_weights:
                bandwidth_unit = available_bandwidth / bandwidth_total_weights
                update_task = False

                for task, weight in sending_weights.items():
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        # Calculate the sending resources, update the task and resource usage, and bandwidth availability
                        sending_resources = round_float(task.required_results_data - task.sending_progress)
                        task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)

                        available_bandwidth = round_float(available_bandwidth - sending_resources)
                        bandwidth_total_weights -= weight

                        update_task = True

                    # Update the sending weights for tasks that havent had resources allocated (this cant happen during the loop with dictionaries)
                sending_weights = {task: weight for task, weight in sending_weights.items() if task not in list(task_resource_usage.keys())}

            if sending_weights:
                bandwidth_unit = round_float(available_bandwidth / bandwidth_total_weights)
                for task, weight in sending_weights.items():
                    sending_resources = round_float(bandwidth_unit * weight)
                    task_resource_usage[task.sending(sending_resources, time_step)] = (task.required_storage, 0, sending_resources)

        return task_resource_usage

    def log_task_resource_usage(self, resource_weights: Dict[Task, float], task_resource_usage: Dict[Task, Tuple[float, float, float]]):
        log.debug(f'{self.name} Server resource usage -> {{' + ', '.join(
            [f'{task.name} Task: ({storage_usage:.3f}, {compute_usage:.3f}, {bandwidth_usage:.3f})'
             for task, (storage_usage, compute_usage, bandwidth_usage) in task_resource_usage.items()]) + '}')
        log.debug(f'{self.name} Server task changes - {{', newline=False)
        for task in resource_weights.keys():
            modified_task = next(_task for _task in task_resource_usage.keys() if task == _task)
            if task.stage is TaskStage.LOADING:
                log.debug(
                    f'{task.name} Task: loading progress {task.loading_progress:.3f} -> {modified_task.loading_progress:.3f} ({modified_task.stage}), ', newline=False)
            elif task.stage is TaskStage.COMPUTING:
                log.debug(f'{task.name} Task: compute progress {task.compute_progress:.3f} -> {modified_task.compute_progress:.3f} ({modified_task.stage}), ', newline=False)
            elif task.stage is TaskStage.SENDING:
                log.debug(
                    f'{task.name} Task: sending progress {task.sending_progress:.3f} -> {modified_task.sending_progress:.3f} ({modified_task.stage}), ', newline=False)
        log.debug('}')
