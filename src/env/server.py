"""
Immutable server class that has storage, computational and storage capacity
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.task import Task
    from typing import Dict, Tuple, List
    

def round_float(value: float) -> float:
    """
    Rounds a number to four decimal places, this is important for the task and server classes in resource allocation

    Args:
        value: The number that is normally a float

    Returns:Rounded number to four decimal places

    """
    return round(value, 4)


class Server(NamedTuple):
    """Server class that takes a name and resource capacity for storage, computation and bandwidth"""

    name: str  # The name of the server

    storage_cap: float        # The server storage capacity
    computational_cap: float  # The server computational capacity
    bandwidth_cap: float      # The server bandwidth capacity

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.computational_cap}, ' \
               f'Bandwidth cap: {self.bandwidth_cap}'

    def __eq__(self, o: object) -> bool:
        # noinspection PyUnresolvedReferences
        return type(o) is Server and o.name == self.name

    def assert_valid(self):
        """
        Assert that the server object is valid
        """
        assert 0 < self.storage_cap and 0 < self.computational_cap and 0 < self.bandwidth_cap

    def allocate_resources(self, resource_weights: Dict[Task, float],
                           time_step: int, error_term: float = 0.1) -> Tuple[List[Task], List[Task]]:
        """
        Allocate resources to tasks by converting a weighting (importance) to an actual resource

        Args:
            resource_weights: A dictionary of task to weighting
            time_step: The current time step
            error_term: The error term to account for rounding effectively

        Returns: Two list, the first being the list of completed or failed task,
                    the second being tasks that are still ongoing

        """
        # Assert that the server tasks are valid
        assert 0 < len(resource_weights)
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or
                   task.stage is TaskStage.SENDING for task in resource_weights.keys())
        assert all(0 <= weight for weight in resource_weights.values())
        for task in resource_weights.keys():
            task.assert_valid()

        # Group the tasks by stage
        loading_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items()
                                              if task.stage is TaskStage.LOADING and 0 < weight}
        compute_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items()
                                              if task.stage is TaskStage.COMPUTING and 0 < weight}
        sending_weights: Dict[Task, float] = {task: weight for task, weight in resource_weights.items()
                                              if task.stage is TaskStage.SENDING and 0 < weight}

        # Resource available, storage is special because some storage is already used due to the previous stage
        available_storage: float = self.storage_cap - sum(task.loading_progress for task in resource_weights.keys())
        available_computation: float = self.computational_cap
        available_bandwidth: float = self.bandwidth_cap

        # Allocate computational resources to the tasks at computing stage
        compute_task_resource_usage = self.allocate_compute_resources(compute_weights, available_computation, time_step)
        # Allocate bandwidth resources (and storage) to the tasks at loading and sending stage
        bandwidth_task_resource_usage = self.allocate_bandwidth_resources(loading_weights, sending_weights,
                                                                          available_storage, available_bandwidth,
                                                                          time_step)
        # If task has weights of zero then allocate resource of only loadings
        no_weights = {
            task._replace(stage=task.has_failed(task.stage, time_step)): (task.loading_progress, 0, 0)
            for task, weight in resource_weights.items() if weight == 0
        }

        # Join the compute and bandwidth resource allocation
        task_resource_usage = {**compute_task_resource_usage, **bandwidth_task_resource_usage, **no_weights}

        # Assert that the updated task are still valid
        for task in task_resource_usage.keys():
            task.assert_valid()
            assert task in list(task_resource_usage.keys())

        # Assert that the resources used are less than available resources
        assert sum(storage_usage for (storage_usage, _, _) in task_resource_usage.values()) <= self.storage_cap + error_term
        assert sum(compute_usage for (_, compute_usage, _) in task_resource_usage.values()) <= self.computational_cap + error_term
        assert sum(bandwidth_usage for (_, _, bandwidth_usage) in task_resource_usage.values()) <= self.bandwidth_cap + error_term

        # Group the updated tasks in those completed or failed and those still ongoing
        unfinished_tasks = [task for task in task_resource_usage.keys()
                            if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
        completed_tasks = [task for task in task_resource_usage.keys()
                           if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]

        return unfinished_tasks, completed_tasks

    @staticmethod
    def allocate_compute_resources(compute_weights: Dict[Task, float], available_computation: float,
                                   time_step: int) -> Dict[Task, Tuple[float, float, float]]:
        """
        Allocate computational resources to tasks

        Args:
            compute_weights: A dictionary of tasks (at computing stage) to weightings
            available_computation: The total available computation (= server.computational_cap)
            time_step: The current time step

        Returns: A dictionary of tasks to their resource usage (storage, compute, bandwidth)

        """

        task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}

        assert all(task.stage is TaskStage.COMPUTING for task in compute_weights.keys())
        assert all(0 < weight for weight in compute_weights.values())

        # It is possible that the percentage of computational resources that could be allocated to a task is greater
        #   than the amount of required computational resource that the task needs to the allocated.
        # So this while loop, checks if the relative maximum computational resources that could be allocated to a
        #   task is less than the required computational resources
        # But in doing this, the relative maximum computational resources that are then available to other tasks has changed,
        #   so this process is looped over till no task will be completed with the relative maximum computational resources.
        task_been_updated: bool = True
        while task_been_updated and compute_weights:
            # Base unit of computational resources relative to the sum of weights for the compute resources
            compute_unit = round_float(available_computation / sum(compute_weights.values()))
            task_been_updated = False

            for task, weight in compute_weights.items():
                # If the weight compute resources are less than the needed computational resources,
                #   allocate only the required resources
                if task.required_computation - task.compute_progress <= weight * compute_unit:
                    compute_resources = round_float(task.required_computation - task.compute_progress)

                    # Set the updated task with the new resources and the resource used by the task
                    updated_task = task.allocate_compute_resources(compute_resources, time_step)
                    assert updated_task.stage is TaskStage.SENDING or updated_task.stage is TaskStage.FAILED
                    task_resource_usage[updated_task] = (task.required_storage, compute_resources, 0)
                    available_computation = round_float(available_computation - compute_resources)
                    task_been_updated = True

            # Update the compute weight based on the task that have had resources allocated
            #   (must be done here as dictionaries cant be used during the loop)
            compute_weights = {task: weight for task, weight in compute_weights.items()
                               if task not in list(task_resource_usage.keys())}

        # If there are any tasks that their compute stage isn't completed using the compute unit
        if compute_weights:
            # The compute unit with the available computational resources leftover
            compute_unit = round_float(available_computation / sum(compute_weights.values()))

            for task, weight in compute_weights.items():
                # Updated the task with the compute resources
                compute_resources = round_float(compute_unit * weight)
                updated_task = task.allocate_compute_resources(compute_resources, time_step)
                task_resource_usage[updated_task] = (task.required_storage, compute_resources, 0)

        return task_resource_usage

    @staticmethod
    def allocate_bandwidth_resources(loading_weights: Dict[Task, float], sending_weights: Dict[Task, float],
                                     available_storage: float, available_bandwidth: float,
                                     time_step: int) -> Dict[Task, Tuple[float, float, float]]:
        """
        Allocate bandwidth (and storage) resources to task at Loading or Sending stages

        Args:
            loading_weights: A dictionary of task (at loading stage) to weights
            sending_weights: A dictionary of task (at sending stage) to weights
            available_storage: The available storage of the server
            available_bandwidth: The available bandwidth of the server
            time_step: The current time step

        Returns: A dictionary of tasks to resources used

        """
        # Task resource usage dictionary of task to resource usage
        task_resource_usage: Dict[Task, Tuple[float, float, float]] = {}

        # Checks that the arguments are valid
        assert all(task.stage is TaskStage.LOADING for task in loading_weights.keys())
        assert all(0 < weight for weight in loading_weights.values())
        assert all(task.stage is TaskStage.SENDING for task in sending_weights.keys())
        assert all(0 < weight for weight in sending_weights.values())

        # Using a similar idea to the compute weight allocation however has four stages to it
        # Stage 1. Tries finding tasks that can finish their current task stage
        tasks_been_updated: List[Task] = [None]
        while tasks_been_updated and (loading_weights or sending_weights):
            # The weighting bandwidth units
            bandwidth_unit = round_float(
                available_bandwidth / (sum(loading_weights.values()) + sum(sending_weights.values())))

            # Reset the list of tasks that have been updated
            tasks_been_updated = []
            # Stage 1.1 - check if sending tasks can be finished
            for task, weight in sending_weights.items():
                if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                    # Calculate the sending resources, update the task and resource usage, and bandwidth availability
                    sending_resources = round_float(task.required_results_data - task.sending_progress)

                    # Update the task and check that the stage is either completed or failed
                    updated_task = task.allocate_sending_resources(sending_resources, time_step)
                    assert updated_task.stage is TaskStage.COMPLETED or updated_task.stage is TaskStage.FAILED

                    # Add resource usage and that the task has been updated
                    task_resource_usage[updated_task] = (task.required_storage, 0, sending_resources)
                    tasks_been_updated.append(updated_task)

                    # Update available bandwidth due to the sending resources
                    available_bandwidth = round_float(available_bandwidth - sending_resources)

            # Update the sending weights by removing tasks that have been updated
            sending_weights = {task: weight for task, weight in sending_weights.items() if
                               task not in tasks_been_updated}

            # Stage 1.2 - Check if loading tasks can be finished
            for task, weight in loading_weights.items():
                # Check that the resources required to complete the loading stage is less than min available resources
                if task.required_storage - task.loading_progress <= min(weight * bandwidth_unit, available_storage,
                                                                        available_bandwidth):
                    # Calculate the loading resources, update the task and resource usage, and bandwidth/storage availability
                    loading_resources = round_float(task.required_storage - task.loading_progress)

                    # Update the task and check that stage is either computing or failed
                    updated_task = task.allocate_loading_resources(loading_resources, time_step)
                    assert updated_task.stage is TaskStage.COMPUTING or updated_task.stage is TaskStage.FAILED

                    # Add resource usage and that the task has been updated
                    task_resource_usage[updated_task] = (task.required_storage, 0, loading_resources)
                    tasks_been_updated.append(updated_task)

                    # Update available storage and bandwidth due to loading resources
                    available_storage = round_float(available_storage - loading_resources)
                    available_bandwidth = round_float(available_bandwidth - loading_resources)

            # Update the loading weights by removing tasks that have been updated
            loading_weights = {task: weight for task, weight in loading_weights.items() if
                               task not in tasks_been_updated}

        # Stage 2 - Try to allocate loading tasks with the maximum available storage/bandwidth resources
        if loading_weights or sending_weights:
            # Total sum of bandwidth weights
            bandwidth_total_weights = sum(loading_weights.values()) + sum(sending_weights.values())

            # Try to allocate resources for loading resources
            for task, weight in loading_weights.items():
                # Calculate the loading resources available to the task
                loading_resources = round_float(
                    min(available_bandwidth / bandwidth_total_weights * weight, available_storage))

                # Update the tasks with the loading resources
                updated_task = task.allocate_loading_resources(loading_resources, time_step)
                # TODO this may not be true because of the available storage
                # assert updated_task.stage is TaskStage.LOADING or updated_task.stage is TaskStage.FAILED

                # Add the resource usage
                task_resource_usage[updated_task] = (updated_task.loading_progress, 0, loading_resources)

                # Update available storage and bandwidth resources due to loading resources and the total weights
                available_storage = round_float(available_storage - loading_resources)
                available_bandwidth = round_float(available_bandwidth - loading_resources)
                bandwidth_total_weights -= weight

            # Reset the variables to allow the while for the first iteration
            tasks_been_updated = [None]
            while tasks_been_updated and sending_weights:
                bandwidth_unit = available_bandwidth / bandwidth_total_weights
                tasks_been_updated = []

                for task, weight in sending_weights.items():
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        # Calculate the sending resources, update the task and resource usage, and bandwidth availability
                        sending_resources = round_float(task.required_results_data - task.sending_progress)

                        # Update the tasks with the sending resources
                        updated_task = task.allocate_sending_resources(sending_resources, time_step)
                        assert updated_task.stage is TaskStage.COMPLETED or updated_task.stage is TaskStage.FAILED

                        # Add resource usage and that the task has been updated
                        task_resource_usage[updated_task] = (task.required_storage, 0, sending_resources)
                        tasks_been_updated.append(updated_task)

                        # Update available bandwidth due to sending resources and update bandwidth weights
                        available_bandwidth = round_float(available_bandwidth - sending_resources)
                        bandwidth_total_weights -= weight

                    # Update the sending weights for tasks that haven't had resources allocated
                    #   (this cant happen during the loop with dictionaries)
                sending_weights = {task: weight for task, weight in sending_weights.items()
                                   if task not in tasks_been_updated}

            # Allocate the remaining resources to the sending tasks
            if sending_weights:
                # Bandwidth units
                bandwidth_unit = round_float(available_bandwidth / bandwidth_total_weights)
                for task, weight in sending_weights.items():
                    # Sending resources
                    sending_resources = round_float(bandwidth_unit * weight)

                    # Update the task
                    updated_task = task.allocate_sending_resources(sending_resources, time_step)
                    # TODO update due to previous stages not using full resources
                    # assert updated_task.stage is TaskStage.SENDING or updated_task.stage is TaskStage.FAILED

                    # Add the resource usage
                    task_resource_usage[updated_task] = (task.required_storage, 0, sending_resources)

        # Return the task resource usage
        return task_resource_usage
