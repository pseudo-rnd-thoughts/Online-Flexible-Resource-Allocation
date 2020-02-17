"""Task class"""

from __future__ import annotations

from typing import Optional, Tuple, List, TYPE_CHECKING

import core.log as log
from core.task_stage import TaskStage

if TYPE_CHECKING:
    from core.server import Server
    from agents.dqn_agent import Trajectory


class Task:
    """
    Task class for encasing required resources and stage progress
    """

    stage: TaskStage = TaskStage.NOT_ASSIGNED  # The current stage that the task is in

    server: Optional[Server] = None  # The allocated server of the task
    price: float  # The price that the server is bought at

    loading_progress: float = 0  # The current progress in loading the task into memory
    compute_progress: float = 0  # The current progress in computing the task
    sending_results_progress: float = 0  # The current progress in sending the results data

    allocated_resources: List[Tuple[str, float, int]] = []

    pricing_trajectory: Optional[Trajectory] = None

    def __init__(self, name: str, auction_time: int, deadline: int,
                 required_storage: int, required_computation: int, required_results_data: int):
        self.name: str = name

        self.auction_time: int = auction_time  # The time point that the task is released for auction
        self.deadline: int = deadline  # The time point that the task must be completed by

        self.required_storage: int = required_storage  # The amount of data required to store the task in memory
        self.required_computation: int = required_computation  # The amount of computation required
        self.required_results_data: int = required_results_data  # The amount of results data to send back

    def __str__(self):
        if self.stage is TaskStage.NOT_ASSIGNED:
            return f'{self.name} Task - Stage: {self.stage}, Auction time: {self.auction_time}, deadline: {self.deadline}, ' \
                   f'storage: {self.required_storage}, comp: {self.required_computation}, results data: {self.required_results_data}'
        elif self.stage is TaskStage.LOADING:
            return f'{self.name} Task - Deadline: {self.deadline}, Loading: {self.loading_progress / self.required_storage:.3f}'
        elif self.stage is TaskStage.COMPUTING:
            return f'{self.name} Task - Deadline: {self.deadline}, Compute: {self.compute_progress / self.required_computation:.3f}'
        elif self.stage is TaskStage.SENDING:
            return f'{self.name} Task - Deadline: {self.deadline}, Sending: {self.sending_results_progress / self.required_results_data:.3f}'
        else:
            return f'{self.name} Task - Stage: {self.stage}, Deadline: {self.deadline}'

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())

    def normalise_task_info(self, server, time_step):
        return [self.required_storage / server.storage_capacity,
                self.required_storage / server.bandwidth_capacity,
                self.required_computation / server.computational_capacity,
                self.required_results_data / server.bandwidth_capacity,

                self.loading_progress / self.required_storage,
                self.compute_progress / self.required_computation,
                self.sending_results_progress / self.required_results_data,
                self.deadline - time_step]

    def allocate_server(self, server: Server, price: float, time_step: int):
        """Allocates a server and price for the task"""

        # Check that the task is not allocated already
        assert self.server is None, \
            "Task {} is already allocated a server {} while trying to allocate {}".format(self.name, self.server.name, server.name)
        assert self.stage is TaskStage.NOT_ASSIGNED, \
            "Task {} stage is {} while trying to allocate a server {}".format(self.name, self.server.name, server.name)

        # Set the server, stage and price variables
        self.server = server
        self.stage = TaskStage.LOADING
        self.price = price
        self.allocated_resources.append(("auctioned", price, time_step))

    def allocate_loading_resources(self, loading_speed: float, time_step: int):
        """
        Allocate the loading speed of the task
        :param loading_speed: The loading speed for the task
        :param time_step: The time step that the resources are allocated
        """
        assert self.server is not None
        assert self.stage is TaskStage.LOADING
        assert 0 <= loading_speed <= self.required_storage - self.loading_progress

        self.loading_progress += loading_speed
        if self.loading_progress == self.required_storage:
            self.stage = TaskStage.COMPUTING

        log.info(f'\t\t{self.name} Task - loading speed {loading_speed}, required loading - {self.required_storage} '
                 f'for progress of {self.loading_progress / self.required_storage:.3}')
        self.allocated_resources.append(('loading', loading_speed, time_step))

    def allocate_compute_resources(self, compute_speed: float, time_step: int):
        """
        Allocate the compute speed of the task
        :param compute_speed: The compute speed for the task
        :param time_step: The time step that the resources are allocated
        """
        assert self.server is not None
        assert self.stage is TaskStage.COMPUTING
        assert 0 <= compute_speed <= self.required_computation - self.compute_progress

        self.compute_progress += compute_speed
        if self.compute_progress == self.required_computation:
            self.stage = TaskStage.SENDING

        log.info(f'\t\t{self.name} Task - computing speed {compute_speed}, required computation - {self.required_computation} '
                 f'for progress of {self.compute_progress / self.required_computation:.3}')
        self.allocated_resources.append(('compute', compute_speed, time_step))

    def allocate_sending_resources(self, sending_speed: float, time_step: int):
        """
        Allocate the sending speed of the task
        :param sending_speed: The sending speed for the task
        :param time_step: The time step that the resources are allocated
        """
        assert self.server is not None
        assert self.stage is TaskStage.SENDING
        assert 0 <= sending_speed <= self.required_results_data - self.sending_results_progress

        self.sending_results_progress += sending_speed
        if self.sending_results_progress == self.required_results_data:
            self.stage = TaskStage.COMPLETED

        log.info(f'\t\t{self.name} Task - sending speed {sending_speed}, required results data - {self.required_results_data} '
                 f'for progress of {self.sending_results_progress / self.required_results_data:.3}')
        self.allocated_resources.append(('sending', sending_speed, time_step))

    def reset(self):
        log.debug('Reset task {}'.format(self.name))

        self.stage = TaskStage.NOT_ASSIGNED
        self.server = None
        self.loading_progress = 0
        self.compute_progress = 0
        self.sending_results_progress = 0
        self.pricing_trajectory = None
