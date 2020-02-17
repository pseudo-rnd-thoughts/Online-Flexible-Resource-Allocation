"""Task stage Enum"""

from enum import Enum, auto


class TaskStage(Enum):
    """
    Enum for the current stage that a task is in
    """
    NOT_ASSIGNED = auto()  # Not server allocated yet
    LOADING = auto()  # Loads the task
    COMPUTING = auto()  # Computes the task
    SENDING = auto()  # Sends the results back
    COMPLETED = auto()  # The task is complete
    INCOMPLETE = auto()  # The task is incomplete within the time limit

    def __str__(self):
        if self is TaskStage.NOT_ASSIGNED:
            return 'Not assigned'
        elif self is TaskStage.LOADING:
            return 'Loading'
        elif self is TaskStage.COMPUTING:
            return 'Computing'
        elif self is TaskStage.SENDING:
            return 'Sending'
        elif self is TaskStage.COMPLETED:
            return 'Completed'
        elif self is TaskStage.INCOMPLETE:
            return 'Incomplete'