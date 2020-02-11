"""Logging of a message to either the console or file"""

from enum import Enum, auto


class DebugLevel(Enum):
    """Debug level enum"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()


debug_filename: str = "log.file"
file_debug_level: DebugLevel = DebugLevel.CRITICAL
console_debug_level: DebugLevel = DebugLevel.INFO


def log_message(message: str, debug_level: DebugLevel, newline: bool = True):
    """
    Debug the message to either console or files
    :param message: The string message
    :param debug_level: The debug level
    :param newline: If to append a new line to end of the message
    """
    if file_debug_level < debug_level:
        with open(debug_filename, 'w') as file:
            if newline:
                file.write(message + '\n')
            else:
                file.write(message)

    if console_debug_level < debug_level:
        if newline:
            print(message)
        else:
            print(message, end="")


def debug(message: str, newline: bool = True):
    log_message(message, DebugLevel.DEBUG, newline)


def info(message: str, newline: bool = True):
    log_message(message, DebugLevel.INFO, newline)


def warning(message: str, newline: bool = True):
    log_message(message, DebugLevel.WARNING, newline)


def critical(message: str, newline: bool = True):
    log_message(message, DebugLevel.CRITICAL, newline)
