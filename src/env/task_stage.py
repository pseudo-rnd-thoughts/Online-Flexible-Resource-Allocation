"""
Task stage Enum
"""

from enum import Enum, auto


class TaskStage(Enum):
    """
    An enum to encode the current stage of a task
    """

    UNASSIGNED = auto()  # Not server allocated yet

    LOADING = auto()  # Loads the task
    COMPUTING = auto()  # Computes the task
    SENDING = auto()  # Sends the results back

    COMPLETED = auto()  # The task is complete
    FAILED = auto()  # The task is incomplete within the time limit
