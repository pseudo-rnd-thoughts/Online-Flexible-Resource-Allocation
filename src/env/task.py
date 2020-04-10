"""
Implementation of a task
"""

from env.task_stage import TaskStage


def round_float(value: float) -> float:
    """
    Rounds a number to four decimal places, this is important for the task and server classes in resource allocation

    Args:
        value: The number that is normally a float

    Returns:Rounded number to four decimal places

    """
    return round(value, 4)


class Task:
    """
    Task class that has a name, price, required resources, auction and deadline time and the progress of the task stages
    """

    def __init__(self, name: str, required_storage: float, required_computation: float, required_results_data: float,
                 auction_time: int, deadline: int, stage: TaskStage = TaskStage.UNASSIGNED,
                 loading_progress: float = 0.0, compute_progress: float = 0.0, sending_progress: float = 0.0,
                 price: float = -1):
        """
        Constructor of the task

        Args:
            name: The name of the task
            required_storage: The required storage of the task
            required_computation: The required computation of the task
            required_results_data: The required results data of the task
            auction_time: The auction time step of the task
            deadline: The deadline time step of the task
            stage: The task stage, initially UNASSIGNED when not assigned
            loading_progress: The loading progress of the task, initially 0 when the task is not assigned
            compute_progress: The compute progress of the task, initially 0 when the task is not assigned or in LOADING stage
            sending_progress: The sending progress of the task, initially 0 when the task is not assigned or in LOADING or COMPUTE stage
            price: The price of the task
        """
        self.name: str = name  # The task name

        self.required_storage: float = required_storage  # The required storage of the task
        self.required_computation: float = required_computation  # The required computation of the task
        self.required_results_data: float = required_results_data  # The required results data of the task

        self.auction_time: int = auction_time  # The auction time for the task
        self.deadline: int = deadline  # The deadline time for the task

        self.stage: TaskStage = stage  # The task sage of the task
        self.loading_progress: float = loading_progress  # The loading progress, once finished this will be equal to the required storage
        self.compute_progress: float = compute_progress  # The compute progress, once finished this will be equal to the required computation
        self.sending_progress: float = sending_progress  # The sending progress, once finished this will be equal to the required results data

        self.price: float = price  # The price of the task, defaults to -1 when not set

        self.assert_valid()

    def assign_server(self, price: float, time_step: int):
        """
        The process of assigning the task to the server

        Args:
            price: The price that the task is won at
            time_step: The time step when the task is won

        Returns: The updated task

        """
        assert 0 < price
        assert self.auction_time == time_step

        self.price = price
        self.stage = TaskStage.LOADING

    def allocate_no_resources(self, time_step: int):
        """
        Don't allocate resources to the tasks

        Args:
            time_step: The current time steps
        """
        assert self.stage is TaskStage.LOADING or self.stage is TaskStage.COMPUTING or self.stage is TaskStage.SENDING

        self.stage = self.has_failed(time_step)

    def allocate_loading_resources(self, loading_resources: float, time_step: int, error_term: float = 0.05):
        """
        The loading of task on a server that increases the loading process with loading_resources at time_step

        Args:
            loading_resources: The loading resources applied to the task
            time_step: The time step that the resources are applied at
            error_term: The error term between the required storage and the loading progress at which to ignore
        """
        assert self.stage is TaskStage.LOADING and self.loading_progress < self.required_storage, str(self)

        self.loading_progress = round_float(self.loading_progress + loading_resources)
        if self.required_storage - self.loading_progress < error_term:
            self.loading_progress = self.required_storage
            self.stage = TaskStage.COMPUTING

        self.stage = self.has_failed(time_step)

    def allocate_compute_resources(self, compute_resources: float, time_step: int, error_term: float = 0.05):
        """
        The computing of the task on a server that increase the computing process with compute_resources at time_step

        Args:
            compute_resources: The compute resources applied to the task
            time_step: The time step that the resources are applied at
            error_term: The error term between the required computation and the compute progress at which to ignore
        """
        assert self.stage is TaskStage.COMPUTING and self.compute_progress < self.required_computation

        self.compute_progress = round_float(self.compute_progress + compute_resources)
        if self.required_computation - self.compute_progress < error_term:
            self.compute_progress = self.required_computation
            self.stage = TaskStage.SENDING

        self.stage = self.has_failed(time_step)

    def allocate_sending_resources(self, sending_resources: float, time_step: int, error_term: float = 0.05):
        """
        The sending of the task on a server that increase the sending process with sending_resources at time_step

        Args:
            sending_resources: The sending resources applied to the task
            time_step: The time step that the resources are applied at
            error_term: The error term between the required results data and the sending progress at which to ignore
        """
        assert self.stage is TaskStage.SENDING and self.sending_progress < self.required_results_data

        self.sending_progress = round_float(self.sending_progress + sending_resources)
        if self.required_results_data - self.sending_progress < error_term:
            self.sending_progress = self.required_results_data
            self.stage = TaskStage.COMPLETED

        self.stage = self.has_failed(time_step)

    def has_failed(self, time_step: int) -> TaskStage:
        """
        Check if the task has failed if the time step is greater than deadline

        Args:
            time_step: The current time step

        Returns: The updated task stage of the task

        """
        assert time_step <= self.deadline

        if self.deadline == time_step and self.stage is not TaskStage.COMPLETED:
            return TaskStage.FAILED
        else:
            return self.stage

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
                        f'Failed {self.name} Task compute progress: {self.compute_progress} < required comp: {self.required_computation}, {str(self)}'
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
            return f'{self.name} Task ({hex(id(self))}) - Unassigned, Storage: {self.required_storage}, Comp: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'

        # The task is assigned to a server with a progress on a stage
        elif self.stage is TaskStage.LOADING:
            return f'{self.name} Task ({hex(id(self))}) - Loading progress ({self.loading_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_computation}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.COMPUTING:
            return f'{self.name} Task ({hex(id(self))}) - Compute progress ({self.compute_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_computation}, Results data: {self.required_results_data}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.SENDING:
            return f'{self.name} Task ({hex(id(self))}) - Sending progress  ({self.sending_progress}), Storage: {self.required_storage}, ' \
                   f'Comp: {self.required_computation}, Results data: {self.required_results_data}, Deadline: {self.deadline}'

        # The task has finished so is completed or failed (due to not being completed within time)
        elif self.stage is TaskStage.COMPLETED:
            return f'{self.name} Task ({hex(id(self))}) - Completed, Storage: {self.required_storage}, Comp: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}'
        elif self.stage is TaskStage.FAILED:
            return f'{self.name} Task ({hex(id(self))}) - Failed, Storage: {self.required_storage}, Comp: {self.required_computation}, ' \
                   f'Results data: {self.required_results_data}, Auction time: {self.auction_time}, Deadline: {self.deadline}, ' \
                   f'Loading ({self.loading_progress}), Computing ({self.compute_progress}), Sending ({self.sending_progress}) progress'

        else:
            raise Exception(f'Unknown stage: {self.stage}')

    def __eq__(self, o: object) -> bool:
        # noinspection PyUnresolvedReferences
        return type(o) is Task and o.name == self.name

    def deep_eq(self, task) -> bool:
        return type(task) is Task and \
               all(getattr(self, attribute) == getattr(task, attribute)
                   for attribute in ['name', 'auction_time', 'deadline', 'price', 'stage',
                                     'required_storage', 'required_computation', 'required_results_data',
                                     'loading_progress', 'compute_progress', 'sending_progress'])
