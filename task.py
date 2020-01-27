"""Task class"""

from enum import Enum, auto

from agent import Agent


class TaskStage(Enum):
    """
    Enum for the current stage that a task is in
    """
    NOT_ASSIGN = auto()  # Not server allocated yet
    LOADING = auto()     # Loads the task
    COMPUTING = auto()   # Computes the task
    SENDING = auto()     # Sends the results back
    COMPLETE = auto()    # The task is complete


class Task(object):
    """
    Task class for encasing required resources and stage progress
    """

    stage: TaskStage = TaskStage.NOT_ASSIGN  # The current stage that the task is in

    allocated_agent: Agent = None  # The allocated agent of the task
    price: int  # The price that the server is bought at

    loading_progress: int = 0  # The current progress in loading the task into memory
    compute_progress: int = 0  # The current progress in computing the task
    sending_results_progress: int = 0  # The current progress in sending the results data

    def __init__(self, name: str, release_time: int, start_time: int, deadline_time: int,
                 required_storage: int, required_computation: int, required_results_data: int,
                 value: int):
        self.name: str = name

        self.release_time: int = release_time  # The time point that the task is released for auction
        self.start_time: int = start_time  # The time point that the task can start loading
        self.deadline_time: int = deadline_time  # The time point that the task must be completed by

        self.required_storage: int = required_storage  # The amount of data required to store the task in memory
        self.required_computation: int = required_computation  # The amount of computation required
        self.required_results_data: int = required_results_data  # The amount of results data to send back

        self.value: int = value  # The private value of the task

    def allocate_server(self, server: Server, price: int):
        """
        Allocates a server and price for the task
        """

        # Check that the task is not allocated already
        assert self.allocated_server is None, \
            "Task {} is already allocated a server {} while trying to allocate {}"\
                .format(self.name, self.allocated_server.name, server.name)

        assert self.stage == TaskStage.NOT_ASSIGN, \
            "Task {} stage is {} while trying to allocate a server {}"\
                .format(self.name, self.allocated_server.name, server.name)

        # Set the server, stage and price variables
        self.allocated_server = server
        self.stage = TaskStage.LOADING
        self.price = price

    def allocate_loading_resources(self, loading_speed: float):
        """
        Allocate the loading speed of the task
        :param loading_speed: The loading speed for the task
        """

        assert self.allocated_server is not None
        assert self.stage == TaskStage.LOADING
        assert 0 <= loading_speed <= self.required_storage - self.loading_progress

        self.loading_progress += loading_speed
        if self.loading_progress == self.required_storage:
            self.stage = TaskStage.COMPUTING

    def allocate_compute_resources(self, compute_speed: float):
        """
        Allocate the compute speed of the task
        :param compute_speed: The compute speed for the task
        """

        assert self.allocated_server is not None
        assert self.stage == TaskStage.COMPUTING
        assert 0 <= compute_speed <= self.required_computation - self.compute_progress

        self.compute_progress += compute_speed
        if self.compute_progress == self.required_computation:
            self.stage = TaskStage.SENDING

    def allocate_sending_resources(self, sending_speed: float):
        """
        Allocate the sending speed of the task
        :param sending_speed: The sending speed for the task
        """

        assert self.allocated_server is not None
        assert self.stage == TaskStage.SENDING
        assert 0 <= sending_speed <= self.required_results_data - self.sending_results_progress

        self.sending_results_progress += sending_speed
        if self.sending_results_progress == self.required_results_data:
            self.stage = TaskStage.COMPLETE
