"""Task class"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.server import Server, round_float
    from typing import List


# noinspection DuplicatedCode
class Task(NamedTuple):
    """
    Task class that has a named, price, required resources, auction and deadline time step and the progress of resources
    """

    name: str

    required_storage: float
    required_comp: float
    required_results_data: float

    auction_time: int
    deadline: int

    stage: TaskStage = TaskStage.UNASSIGNED
    loading_progress: float = 0
    compute_progress: float = 0
    sending_progress: float = 0

    price: float = -1

    def loading(self, loading_resources, time_step: int) -> Task:
        assert self.stage is TaskStage.LOADING and self.loading_progress < self.required_storage
        updated_loading_progress = round_float(self.loading_progress + loading_resources)
        updated_stage = TaskStage.LOADING if updated_loading_progress < self.required_storage else TaskStage.COMPUTING
        return self._replace(loading_progress=updated_loading_progress, stage=self.has_failed(updated_stage, time_step))

    def compute(self, compute_resources, time_step: int) -> Task:
        assert self.stage is TaskStage.COMPUTING and self.compute_progress < self.required_comp
        updated_compute_progress = round_float(self.compute_progress + compute_resources)
        updated_stage = TaskStage.COMPUTING if updated_compute_progress < self.required_comp else TaskStage.SENDING
        return self._replace(compute_progress=updated_compute_progress, stage=self.has_failed(updated_stage, time_step))

    def sending(self, sending_resources, time_step: int) -> Task:
        assert self.stage is TaskStage.SENDING and self.sending_progress < self.required_results_data
        updated_sending_progress = round_float(self.sending_progress + sending_resources)
        updated_stage = TaskStage.SENDING if updated_sending_progress < self.required_results_data else TaskStage.COMPLETED
        return self._replace(sending_progress=updated_sending_progress, stage=self.has_failed(updated_stage, time_step))

    def has_failed(self, updated_stage: TaskStage, time_step: int) -> TaskStage:
        assert time_step <= self.deadline
        if self.deadline == time_step and updated_stage is not TaskStage.COMPLETED:
            return TaskStage.FAILED
        else:
            return updated_stage

    def assert_valid(self):
        if self.stage is not TaskStage.FAILED:
            if self.stage is TaskStage.LOADING:
                assert self.loading_progress < self.required_storage and self.compute_progress == 0 and self.sending_progress == 0
            else:
                assert self.required_storage <= self.loading_progress
                if self.stage is TaskStage.COMPUTING:
                    assert self.compute_progress < self.required_comp, \
                        f'Failed {self.name} Task compute progress: {self.compute_progress} < required comp: {self.required_comp}, {str(self)}'
                    assert self.sending_progress == 0, \
                        f'Failed {self.name} Task is Computing but has sending progress: {self.sending_progress}'
                else:
                    assert self.required_comp <= self.compute_progress
                    if self.stage is TaskStage.SENDING:
                        assert self.sending_progress < self.required_results_data
                    else:
                        assert self.stage is TaskStage.COMPLETED

    def __str__(self) -> str:
        # The task is unassigned therefore there is no progress on stages
        if self.stage is TaskStage.UNASSIGNED:
            return f'{self.name} Task ({hex(id(self))}) - Unassigned, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'

        # The task is assigned to a server with a progress on a stage
        elif self.stage is TaskStage.LOADING:
            return f'{self.name} Task ({hex(id(self))}) - Loading progress ({self.loading_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.COMPUTING:
            return f'{self.name} Task ({hex(id(self))}) - Compute progress ({self.compute_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.SENDING:
            return f'{self.name} Task ({hex(id(self))}) - Sending progress  ({self.sending_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'

        # The task has finished so is completed or failed (due to not being completed within time)
        elif self.stage is TaskStage.COMPLETED:
            return f'{self.name} Task ({hex(id(self))}) - Completed, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.INCOMPLETE:
            return f'{self.name} Task ({hex(id(self))}) - Failed, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'

    def __eq__(self, o: object) -> bool:
        # noinspection PyUnresolvedReferences
        return type(o) is Task and o.name == self.name
