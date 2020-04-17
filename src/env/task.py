"""Task class"""

from __future__ import annotations

from typing import NamedTuple

from env.server import round_float
from env.task_stage import TaskStage


class Task(NamedTuple):
    """
    Task class that has a named, price, required resources, auction and deadline time step and the progress of resources
    """

    name: str

    required_storage: float
    required_computation: float
    required_results_data: float

    auction_time: int
    deadline: int

    stage: TaskStage = TaskStage.UNASSIGNED
    loading_progress: float = 0.0
    compute_progress: float = 0.0
    sending_progress: float = 0.0

    price: float = -1

    def assign_server(self, price: float, time_step: int) -> Task:
        """
        The process of assigning the task to the server

        Args:
            price: The price that the task is won at
            time_step: The time step when the task is won

        Returns: The updated task

        """
        assert 0 < price
        assert self.auction_time == time_step
        return self._replace(price=price, stage=TaskStage.LOADING)

    def allocate_loading_resources(self, loading_resources: float, time_step: int) -> Task:
        """
        The loading of task on a server that increases the loading process with loading_resources at time_step

        Args:
            loading_resources: The loading resources applied to the task
            time_step: The time step that the resources are applied at

        Returns: The updated task

        """
        assert self.stage is TaskStage.LOADING and self.loading_progress < self.required_storage
        updated_loading_progress = round_float(self.loading_progress + loading_resources)
        updated_stage = TaskStage.LOADING if updated_loading_progress < self.required_storage else TaskStage.COMPUTING
        return self._replace(loading_progress=updated_loading_progress, stage=self.has_failed(updated_stage, time_step))

    def allocate_compute_resources(self, compute_resources: float, time_step: int) -> Task:
        """
        The computing of the task on a server that increase the computing process with compute_resources at time_step

        Args:
            compute_resources: The compute resources applied to the task
            time_step: The time step that the resources are applied at

        Returns: The updated task

        """
        assert self.stage is TaskStage.COMPUTING and self.compute_progress < self.required_computation
        updated_compute_progress = round_float(self.compute_progress + compute_resources)
        updated_stage = TaskStage.COMPUTING if updated_compute_progress < self.required_computation else TaskStage.SENDING
        return self._replace(compute_progress=updated_compute_progress, stage=self.has_failed(updated_stage, time_step))

    def allocate_sending_resources(self, sending_resources: float, time_step: int) -> Task:
        """
        The sending of the task on a server that increase the sending process with sending_resources at time_step

        Args:
            sending_resources: The sending resources applied to the task
            time_step: The time step that the resources are applied at

        Returns: The updated task

        """
        assert self.stage is TaskStage.SENDING and self.sending_progress < self.required_results_data
        updated_sending_progress = round_float(self.sending_progress + sending_resources)
        updated_stage = TaskStage.SENDING if updated_sending_progress < self.required_results_data else TaskStage.COMPLETED
        return self._replace(sending_progress=updated_sending_progress, stage=self.has_failed(updated_stage, time_step))

    def has_failed(self, updated_stage: TaskStage, time_step: int) -> TaskStage:
        """
        Check if the task has failed if the time step is greater than deadline

        Args:
            updated_stage: The current stage of the task
            time_step: The current time step

        Returns: The updated task stage of the task

        """
        assert time_step <= self.deadline
        if self.deadline == time_step and updated_stage is not TaskStage.COMPLETED:
            return TaskStage.FAILED
        else:
            return updated_stage

    def assert_valid(self):
        """
        Assert if the task is valid
        """
        assert self.name != ''
        assert 0 < self.required_storage and 0 < self.required_computation and 0 < self.required_results_data
        assert self.auction_time < self.deadline

        if self.stage is TaskStage.UNASSIGNED:
            assert self.loading_progress == self.compute_progress == self.sending_progress == 0
            assert self.price == -1
        else:
            assert 0 < self.price
            if self.stage is TaskStage.LOADING:
                assert self.loading_progress < self.required_storage
                assert self.compute_progress == self.sending_progress == 0
            else:
                assert self.required_storage <= self.loading_progress, str(self)
                if self.stage is TaskStage.COMPUTING:
                    assert self.compute_progress < self.required_computation, \
                        f'Failed {self.name} Task compute progress: {self.compute_progress} < ' \
                        f'required comp: {self.required_computation}, {str(self)}'
                    assert self.sending_progress == 0, \
                        f'Failed {self.name} Task is Computing but has sending progress: {self.sending_progress}'
                else:
                    assert self.required_computation <= self.compute_progress
                    if self.stage is TaskStage.SENDING:
                        assert self.sending_progress < self.required_results_data
                    else:
                        if self.stage is TaskStage.COMPLETED:
                            assert self.required_results_data <= self.sending_progress
                        else:
                            assert self.stage is TaskStage.FAILED

    def __str__(self) -> str:
        # The task is unassigned therefore there is no progress on stages
        if self.stage is TaskStage.UNASSIGNED:
            return f'{self.name} Task ({hex(id(self))}) - Unassigned, Storage: {self.required_storage}, ' \
                   f'Computation: {self.required_computation}, Results data: {self.required_results_data}, ' \
                   f'Auction time: {self.auction_time}, Deadline: {self.deadline}'

        # The task is assigned to a server with a progress on a stage
        elif self.stage is TaskStage.LOADING:
            return f'{self.name} Task ({hex(id(self))}) - Loading progress ({self.loading_progress}), ' \
                   f'Storage: {self.required_storage}, Computation: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.COMPUTING:
            return f'{self.name} Task ({hex(id(self))}) - Compute progress ({self.compute_progress}), ' \
                   f'Storage: {self.required_storage}, Computation: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.SENDING:
            return f'{self.name} Task ({hex(id(self))}) - Sending progress  ({self.sending_progress}), ' \
                   f'Storage: {self.required_storage}, Computation: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Deadline: {self.deadline}'

        # The task has finished so is completed or failed (due to not being completed within time)
        elif self.stage is TaskStage.COMPLETED:
            return f'{self.name} Task ({hex(id(self))}) - Completed, Storage: {self.required_storage}, ' \
                   f'Computation: {self.required_computation}, Results data: {self.required_results_data}, ' \
                   f'Auction time: {self.auction_time}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.INCOMPLETE:
            return f'{self.name} Task ({hex(id(self))}) - Failed, Storage: {self.required_storage}, ' \
                   f'Computation: {self.required_computation}, Results data: {self.required_results_data}, ' \
                   f'Auction time: {self.auction_time}, Deadline: {self.deadline}'

    def __eq__(self, o: object) -> bool:
        # noinspection PyUnresolvedReferences
        return type(o) is Task and o.name == self.name
