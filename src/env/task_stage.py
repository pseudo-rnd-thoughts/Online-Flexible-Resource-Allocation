from enum import Enum, auto


class TaskStage(Enum):
    UNASSIGNED = auto()  # Server isn't allocated yet

    LOADING = auto()     # At loading stage for the task
    COMPUTING = auto()   # At compute stage for the task
    SENDING = auto()     # At sending stage for the task

    COMPLETED = auto()   # Task is completed successfully
    FAILED = auto()      # Task is incomplete within the deadline
