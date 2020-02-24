"""Task class"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING, List

from env.task_stage import TaskStage

if TYPE_CHECKING:
    from env.server import Server


class Task(NamedTuple):
    """
    Task class that has a named, price, required resources, auction and deadline time step and the progress of resources
    """

    name: str

    required_storage: int
    required_comp: int
    required_results_data: int

    auction_time: int
    deadline: int

    stage: TaskStage = TaskStage.UNASSIGNED
    loading_progress: float = 0
    compute_progress: float = 0
    sending_progress: float = 0

    price: float = -1

    def normalise(self, server: Server, time_step: int) -> List[float]:
        return [self.required_storage / server.storage_cap, self.required_storage / server.bandwidth_cap,
                self.required_comp / server.comp_cap, self.required_results_data / server.bandwidth_cap,
                self.deadline - time_step, self.loading_progress, self.compute_progress, self.sending_progress]

    def loading(self, loading_resources, time_step: int) -> Task:
        updated_loading_progress = self.loading_progress + loading_resources
        updated_stage = TaskStage.LOADING if updated_loading_progress < self.required_storage else TaskStage.COMPUTING
        updated_stage = updated_stage if time_step < self.deadline else TaskStage.FAILED
        return self._replace(loading_progress=updated_loading_progress, stage=updated_stage)

    def compute(self, compute_resources, time_step: int) -> Task:
        updated_compute_progress = self.compute_progress + compute_resources
        updated_stage = TaskStage.COMPUTING if updated_compute_progress < self.required_comp else TaskStage.SENDING
        updated_stage = updated_stage if time_step < self.deadline else TaskStage.FAILED
        return self._replace(compute_progress=updated_compute_progress, stage=updated_stage)

    def sending(self, sending_resources, time_step: int) -> Task:
        updated_sending_progress = self.sending_progress + sending_resources
        updated_stage = TaskStage.SENDING if updated_sending_progress < self.required_results_data else TaskStage.COMPLETED
        updated_stage = TaskStage.FAILED if time_step == self.deadline and updated_stage is not TaskStage.COMPLETED else updated_stage
        return self._replace(sending_progress=updated_sending_progress, stage=updated_stage)

    def __str__(self) -> str:
        # The task is unassigned therefore there is no progress on stages
        if self.stage is TaskStage.UNASSIGNED:
            return f'{self.name} Task ({hex(id(self))}) - Unassigned, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'

        # The task is assigned to a server with a progress on a stage
        elif self.stage is TaskStage.LOADING:
            return f'{self.name} Task ({hex(id(self))}) - Loading progress ({self.loading_progress / self.required_storage}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.COMPUTING:
            return f'{self.name} Task ({hex(id(self))}) - Compute progress ({self.compute_progress / self.required_comp}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.SENDING:
            return f'{self.name} Task ({hex(id(self))}) - Sending progress  ({self.sending_progress / self.required_results_data}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_comp}, Results data: {self.required_results_data}, Deadline: {self.deadline}'

        # The task has finished so is completed or failed (due to not being completed within time)
        elif self.stage is TaskStage.COMPLETED:
            return f'{self.name} Task ({hex(id(self))}) - Completed, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.INCOMPLETE:
            return f'{self.name} Task ({hex(id(self))}) - Failed, Storage: {self.required_storage}, Comp: {self.required_comp}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'

    def __eq__(self, o: object) -> bool:
        return type(o) is Task and o.name == self.name
