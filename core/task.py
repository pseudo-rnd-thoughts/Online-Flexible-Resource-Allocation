"""Task class"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional, Tuple, List, TYPE_CHECKING

import core.log as log

if TYPE_CHECKING:
    from core.server import Server
    from agents.dqn_agent import Trajectory


class TaskStage(Enum):
    """
    Enum for the current stage that a task is in
    """
    NOT_ASSIGN = auto()  # Not server allocated yet
    LOADING = auto()  # Loads the task
    COMPUTING = auto()  # Computes the task
    SENDING = auto()  # Sends the results back
    COMPLETE = auto()  # The task is complete
    INCOMPLETE = auto()  # The task is incomplete within the time limit


class Task:
    """
    Task class for encasing required resources and stage progress
    """

    stage: TaskStage = TaskStage.NOT_ASSIGN  # The current stage that the task is in

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
        if self.stage == TaskStage.NOT_ASSIGN:
            return f'{self.name} Task - stage: {self.stage}, auction time: {self.auction_time}, deadline: {self.deadline}, ' \
                   f'storage: {self.required_storage}, comp: {self.required_computation}, results data: {self.required_results_data}'
        elif self.stage == TaskStage.LOADING:
            return f'{self.name} Task - auction time: {self.auction_time}, deadline: {self.deadline}, loading: {self.loading_progress / self.required_storage:.3f}'
        elif self.stage == TaskStage.COMPUTING:
            return f'{self.name} Task - auction time: {self.auction_time}, deadline: {self.deadline}, compute: {self.compute_progress / self.required_computation:.3f}'
        elif self.stage == TaskStage.SENDING:
            return f'{self.name} Task - auction time: {self.auction_time}, deadline: {self.deadline}, sending: {self.sending_results_progress / self.required_results_data:.3f}'
        else:
            return f'{self.name} Task - stage: {self.stage}, auction time: {self.auction_time}, deadline: {self.deadline}'

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
            "Task {} is already allocated a server {} while trying to allocate {}".format(self.name, self.server.name,
                                                                                          server.name)
        assert self.stage == TaskStage.NOT_ASSIGN, \
            "Task {} stage is {} while trying to allocate a server {}".format(self.name, self.server.name, server.name)

        # Set the server, stage and price variables
        self.server = server
        self.stage = TaskStage.LOADING
        self.price = price

        log.info('\t\tTask {} allocate server {} for price {}'.format(self.name, server.name, price))
        self.allocated_resources.append(("auctioned", price, time_step))

    def allocate_loading_resources(self, loading_speed: float, time_step: int):
        """
        Allocate the loading speed of the task
        :param loading_speed: The loading speed for the task
        :param time_step: The time step that the resources are allocated
        """

        assert self.server is not None
        assert self.stage == TaskStage.LOADING
        assert 0 <= loading_speed <= self.required_storage - self.loading_progress

        self.loading_progress += loading_speed
        if self.loading_progress == self.required_storage:
            self.stage = TaskStage.COMPUTING

        log.info('\t\tTask {} loading resources {} to percent {} and stage {}'
                 .format(self.name, loading_speed, self.loading_progress / self.required_storage, self.stage))
        self.allocated_resources.append(('loading', loading_speed, time_step))

    def allocate_compute_resources(self, compute_speed: float, time_step: int):
        """
        Allocate the compute speed of the task
        :param compute_speed: The compute speed for the task
        :param time_step: The time step that the resources are allocated
        """

        assert self.server is not None
        assert self.stage == TaskStage.COMPUTING
        assert 0 <= compute_speed <= self.required_computation - self.compute_progress

        self.compute_progress += compute_speed
        if self.compute_progress == self.required_computation:
            self.stage = TaskStage.SENDING

        log.info('\t\tTask {} compute resources {} to percent {} and stage {}'
                 .format(self.name, compute_speed, self.compute_progress / self.required_computation, self.stage))
        self.allocated_resources.append(('compute', compute_speed, time_step))

    def allocate_sending_resources(self, sending_speed: float, time_step: int):
        """
        Allocate the sending speed of the task
        :param sending_speed: The sending speed for the task
        :param time_step: The time step that the resources are allocated
        """

        assert self.server is not None
        assert self.stage == TaskStage.SENDING
        assert 0 <= sending_speed <= self.required_results_data - self.sending_results_progress

        self.sending_results_progress += sending_speed
        if self.sending_results_progress == self.required_results_data:
            self.stage = TaskStage.COMPLETE

        log.info('\t\tTask {} sending resources {} to percent {} and stage {}'
                 .format(self.name, sending_speed, self.sending_results_progress / self.required_results_data,
                         self.stage))
        self.allocated_resources.append(('sending', sending_speed, time_step))

    def reset(self):
        log.debug('Reset task {}'.format(self.name))

        self.stage = TaskStage.NOT_ASSIGN
        self.server = None
        self.loading_progress = 0
        self.compute_progress = 0
        self.sending_results_progress = 0
        self.pricing_trajectory = None
