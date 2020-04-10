"""
Implementation of a server
"""

from typing import Dict, Tuple, List

from env.task import Task, round_float
from env.task_stage import TaskStage


class Server:
    """Server class that takes a name and resource capacity for storage, computation and bandwidth"""

    def __init__(self, name: str, storage_cap: float, computational_comp: float, bandwidth_cap: float):
        self.name = name  # The name of the server

        self.storage_cap = storage_cap                # The server storage capacity
        self.computational_comp = computational_comp  # The server computational capacity
        self.bandwidth_cap = bandwidth_cap            # The server bandwidth capacity

    def __str__(self) -> str:
        return f'{self.name} Server - Storage cap: {self.storage_cap}, Comp cap: {self.computational_comp}, ' \
               f'Bandwidth cap: {self.bandwidth_cap}'

    def allocate_resources(self, resource_weights: List[Tuple[Task, float]],
                           time_step: int, error_term: float = 0.05) -> Tuple[List[Task], List[Task]]:
        """
        Allocate resources to tasks by converting a weighting (importance) to an actual resource

        Args:
            resource_weights: A list of tuples of task ands weightings (this cant be a dictionary as task is unhashable)
            time_step: The current time step
            error_term: The error term to account for rounding effectively (for checking on the resources allocated)

        Returns: Two list, the first being the list of completed or failed task,
                    the second being tasks that are still ongoing

        """

        # Assert that all tasks are at the correct stage
        assert all(task.stage is TaskStage.LOADING or task.stage is TaskStage.COMPUTING or task.stage is TaskStage.SENDING
                   for task, weight in resource_weights)
        # Assert that all weights are greater than zero
        assert all(0 <= weight for task, weight in resource_weights)

        # Group the task resource weights by the task stages
        loading_weights: List[Tuple[Task, float]] = [
            (task, weight) for task, weight in resource_weights if task.stage is TaskStage.LOADING and 0 < weight
        ]
        compute_weights: List[Tuple[Task, float]] = [
            (task, weight) for task, weight in resource_weights if task.stage is TaskStage.COMPUTING and 0 < weight
        ]
        sending_weights: List[Tuple[Task, float]] = [
            (task, weight) for task, weight in resource_weights if task.stage is TaskStage.SENDING and 0 < weight
        ]

        # Resource available, storage is special because some storage is already used due to the previous stage
        available_storage: float = self.storage_cap - sum(task.loading_progress for task, weight in resource_weights)
        available_computation: float = self.computational_comp
        available_bandwidth: float = self.bandwidth_cap

        # Allocate computational resources to the tasks at computing stage
        compute_task_resource_usage = self.allocate_compute_resources(compute_weights, available_computation, time_step)
        # Allocate bandwidth resources (and storage) to the tasks at loading and sending stage
        bandwidth_task_resource_usage = self.allocate_bandwidth_resources(loading_weights, sending_weights,
                                                                          available_storage, available_bandwidth,
                                                                          time_step)
        # If task has weights of zero then allocate resource of only loadings

        no_weights: List[Tuple[Task, float, float, float]] = []
        for task, weight in resource_weights:
            if weight == 0:
                task.allocate_no_resources(time_step)
                no_weights.append((task, task.loading_progress, 0.0, 0.0))

        # Join the compute and bandwidth resource allocation
        task_resource_usage = compute_task_resource_usage + bandwidth_task_resource_usage + no_weights

        # Assert that the updated task are still valid
        for task, _, _, _ in task_resource_usage:
            task.assert_valid()
            assert task in [task for task, weight in resource_weights]
        # Assert that the resources used are less than available resources
        assert sum(storage_usage for _, storage_usage, _, _ in task_resource_usage) <= self.storage_cap + error_term
        assert sum(compute_usage for _, _, compute_usage, _ in task_resource_usage) <= self.computational_comp + error_term
        assert sum(bandwidth_usage for _, _, _, bandwidth_usage in task_resource_usage) <= self.bandwidth_cap + error_term

        # Group the updated tasks in those completed or failed and those still ongoing
        unfinished_tasks = [task for task, _, _, _ in task_resource_usage
                            if not (task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED)]
        completed_tasks = [task for task, _, _, _ in task_resource_usage
                           if task.stage is TaskStage.COMPLETED or task.stage is TaskStage.FAILED]

        return unfinished_tasks, completed_tasks

    @staticmethod
    def allocate_compute_resources(compute_weights: List[Tuple[Task, float]], available_computation: float,
                                   time_step: int) -> List[Tuple[Task, float, float, float]]:
        """
        Allocate computational resources to tasks

        Args:
            compute_weights: A list of tuples of tasks (at computing stage) and a weightings
            available_computation: The total available computation (== server.computational_cap)
            time_step: The current time step

        Returns: A list of tuples of tasks to their resource usage (storage, compute, bandwidth)

        """
        # The resources allocated to each task, storage, computation, bandwidth
        task_resources_allocated: List[Tuple[Task, float, float, float]] = []

        # Assert that the compute weights are valid
        assert all(task.stage is TaskStage.COMPUTING for task, weight in compute_weights)
        assert all(0 < weight for task, weight in compute_weights)

        # The aim to allocate as much of the available compute resources to the tasks by iteratively seeing if any
        #   task to overuse the weighted resources allocated. For these tasks, allocate the exact amount of resources
        #   required to finished the compute stage of the task. As a result of this, the compute weight unit will be
        #   increased that as a result could mean that more resources are available for the other tasks. This is
        #   iteratively done till no task's compute stage can be completed with the weighted resources.
        # All remaining tasks have their weighted resources directly added to them.
        task_compute_stage_updated = True
        while task_compute_stage_updated and compute_weights:
            # Base unit of computational resources relative to the sum of weights for the compute resources
            compute_unit = round_float(available_computation / sum(weight for task, weight in compute_weights))
            task_compute_stage_updated = False

            # Loop over all of the tasks to check if the weighted resources is greater than required resources to
            #   complete compute stage
            for task, weight in compute_weights:
                if task.required_computation - task.compute_progress <= weight * compute_unit:
                    # Calculated the required resources instead
                    compute_resources = round_float(task.required_computation - task.compute_progress)

                    # Update the task given the compute resources
                    task.allocate_compute_resources(compute_resources, time_step)
                    assert task.stage == TaskStage.SENDING
                    # Add the task with its resource usage
                    task_resources_allocated.append((task, task.required_storage, compute_resources, 0.0))
                    # As a task has been finished, set the task compute stage updated variable to true
                    task_compute_stage_updated = True

                    # Reduce the available computational resources by the used compute resources
                    available_computation = round_float(available_computation - compute_resources)

            # Update the compute weights with the tasks that haven't had resources allocated to sending stage
            compute_weights = [(task, weight) for task, weight in compute_weights if task.stage is TaskStage.COMPUTING]

        # If there are any tasks that their compute stage isn't completed using the compute unit
        if compute_weights:
            # The compute unit with the available computational resources leftover
            compute_unit = round_float(available_computation / sum(weight for task, weight in compute_weights))

            for task, weight in compute_weights:
                # Calculate the weighted compute resources
                compute_resources = round_float(compute_unit * weight)
                # Allocate the compute resources to the task
                task.allocate_compute_resources(compute_resources, time_step)
                assert task.stage is TaskStage.COMPUTING
                # Add the task with its resources
                task_resources_allocated.append((task, task.required_storage, compute_resources, 0.0))

        return task_resources_allocated

    @staticmethod
    def allocate_bandwidth_resources(loading_weights: List[Tuple[Task, float]], sending_weights: List[Tuple[Task, float]],
                                     available_storage: float, available_bandwidth: float,
                                     time_step: int) -> List[Tuple[Task, float, float, float]]:
        """
        Allocate bandwidth (and storage) resources to task at the Loading or Sending stages

        Args:
            loading_weights: A list of tuples of task (at loading stage) with weights
            sending_weights: A list of tuples of task (at sending stage) with weights
            available_storage: The available storage of the server
            available_bandwidth: The available bandwidth of the server
            time_step: The current time step

        Returns: A list of tasks with the resources used

        """
        # The resources allocated to each task, storage, computation and bandwidth
        task_resource_usage: List[Tuple[Task, float, float, float]] = []

        # Assert that the loading and sending weights are valid
        assert all(task.stage is TaskStage.LOADING for task, weight in loading_weights)
        assert all(0 < weight for task, weight in loading_weights)
        assert all(task.stage is TaskStage.SENDING for task, weight in sending_weights)
        assert all(0 < weight for task, weight in sending_weights)

        # This function is similar in function to the allocate_compute_resources function however with the additional
        #   difficulty of both allocating bandwidth and storage resources at the same time. Because of this, we try
        #   to finish the sending tasks first as it will free up the storage resource next.
        task_stage_updated = True
        while task_stage_updated and (loading_weights or sending_weights):
            # Calculate the bandwidth unit by summing the weights of both the loading and sending weights
            bandwidth_unit = round_float(available_bandwidth / sum(weight for task, weight in loading_weights + sending_weights))
            task_stage_updated = False

            # Loop over all of the tasks to check if the weighted resource is greater than required resource to
            #   complete sending stage
            for task, weight in sending_weights:
                if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                    # Calculate the required resources instead
                    sending_resources = round_float(task.required_results_data - task.sending_progress)

                    # Update the task given the sending resources
                    task.allocate_sending_resources(sending_resources, time_step)
                    assert task.stage is TaskStage.COMPLETED
                    # Add the task with its resource usage
                    task_resource_usage.append((task, task.required_storage, 0, sending_resources))
                    # As a task has been finished, set the task stage updated variable to true
                    task_stage_updated = True

                    # Reduce the available bandwidth resource by the used sending resources
                    available_bandwidth = round_float(available_bandwidth - sending_resources)

            # Update the sending weights with the tasks that haven't had resources allocated
            sending_weights = [(task, weight) for task, weight in sending_weights if task.stage is TaskStage.SENDING]

            # The repeat the same process for tasks being loaded
            for task, weight in loading_weights:
                # Check that the resources required to complete the loading stage is less than min available resources
                if task.required_storage - task.loading_progress <= min(weight * bandwidth_unit, available_storage, available_bandwidth):
                    # Calculate the required resource instead
                    loading_resources = round_float(task.required_storage - task.loading_progress)

                    # Update hte task given the loading resources
                    task.allocate_loading_resources(loading_resources, time_step)
                    assert task.stage is TaskStage.COMPUTING
                    # Add the task with its resource usage
                    task_resource_usage.append((task, task.required_storage, 0, loading_resources))
                    # As a task has been finished, set the task  stage update variable to true
                    task_stage_updated = True

                    # Reduce the available storage and bandwidth resource by the used loading resources
                    available_storage = round_float(available_storage - loading_resources)
                    available_bandwidth = round_float(available_bandwidth - loading_resources)

            # Update the loading weights with the tasks that haven't had resources allocated
            loading_weights = [(task, weight) for task, weight in loading_weights if task.stage is TaskStage.LOADING]

        # If there are any tasks left then allocate the remaining tasks
        if loading_weights or sending_weights:
            # Calculate the total weights of the remaining loading and sending tasks
            bandwidth_total_weights = sum(weight for task, weight in loading_weights + sending_weights)

            # Try to allocate resources for loading tasks, recalculating the bandwidth unit each time
            for task, weight in loading_weights:
                # Calculate the loading resources available to the task
                loading_resources = round_float(min(round_float(available_bandwidth / bandwidth_total_weights * weight), available_storage))

                # Update the task with the loading resources
                task.allocate_loading_resources(loading_resources, time_step)
                assert task.stage is TaskStage.LOADING
                # Add the task with its resource usage
                task_resource_usage.append((task, task.loading_progress, 0, loading_resources))

                # Reduce the available storage and bandwidth resource by the used loading resources
                available_storage = round_float(available_storage - loading_resources)
                available_bandwidth = round_float(available_bandwidth - loading_resources)
                # Update the total weights that have not been allocated
                bandwidth_total_weights -= weight
            # All of the loading tasks have been allocated at this point

            # As not all of the bandwidth resources have been allocated then it may be not possible that a sending task
            #   could be finished with the remaining resources. Therefore rerun the allocate process again
            task_stage_updated = True
            while task_stage_updated and sending_weights:
                # The bandwidth unit
                bandwidth_unit = available_bandwidth / bandwidth_total_weights
                task_stage_updated = False

                # Loop over each of the tasks
                for task, weight in sending_weights:
                    if task.required_results_data - task.sending_progress <= weight * bandwidth_unit:
                        # Calculate the sending resources, update the task and resource usage, and bandwidth availability
                        sending_resources = round_float(task.required_results_data - task.sending_progress)
                        # Update the task with the sending resources
                        task.allocate_sending_resources(sending_resources, time_step)
                        assert task.stage is TaskStage.COMPLETED
                        # Add the task with its resource usage
                        task_resource_usage.append((task, task.required_storage, 0, sending_resources))
                        # As a task has been finished, set the task  stage update variable to true
                        task_stage_updated = True

                        # Reduce the available bandwidth resource by the used sending resources
                        available_bandwidth = round_float(available_bandwidth - sending_resources)
                        # Update the total weights that have not been allocated
                        bandwidth_total_weights -= weight

                    # Update the sending weights with the tasks that haven't had resources allocated
                    sending_weights = [(task, weight) for task, weight in sending_weights
                                       if task.stage is TaskStage.SENDING]

            assert bandwidth_total_weights == sum(weight for task, weight in sending_weights)
            # Otherwise allocate the remaining resources
            if sending_weights:
                # The bandwidth units
                bandwidth_unit = round_float(available_bandwidth / bandwidth_total_weights)
                for task, weight in sending_weights:
                    sending_resources = round_float(bandwidth_unit * weight)
                    # Update the task with the sending resources
                    task.allocate_sending_resources(sending_resources, time_step)
                    assert task.stage is TaskStage.SENDING
                    # Add the task with its resource usage
                    task_resource_usage.append((task, task.required_storage, 0, sending_resources))

        return task_resource_usage
