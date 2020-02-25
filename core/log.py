"""Logging of a message to either the console or file"""

from __future__ import annotations

from enum import Enum


class LogLevel(Enum):
    """Debug level enum"""
    NEURAL_NETWORK = 0
    DEBUG = 1
    INFO = 2
    WARNING = 3
    CRITICAL = 4


debug_filename: str = 'debug.log'
file_debug_level: LogLevel = LogLevel.DEBUG
console_debug_level: LogLevel = LogLevel.CRITICAL


def log_message(message: str, debug_level: LogLevel, newline: bool = True):
    """
    Debug the message to either console or files
    :param message: The string message
    :param debug_level: The debug level
    :param newline: If to append a new line to end of the message
    """
    if file_debug_level.value <= debug_level.value:
        with open(debug_filename, 'a') as file:
            if newline:
                file.write(message + '\n')
            else:
                file.write(message)

    if console_debug_level.value <= debug_level.value:
        if newline:
            print(message)
        else:
            print(message, end="")


def debug(message: str, newline: bool = True):
    """
    Log message at debug level
    :param message: The message
    :param newline: If to have a newline at the end of the message
    """
    log_message(message, LogLevel.DEBUG, newline)


def info(message: str, newline: bool = True):
    """
    Log message at info level
    :param message: The message
    :param newline: If to have a newline at the end of the message
    """
    log_message(message, LogLevel.INFO, newline)


def warning(message: str, newline: bool = True):
    """
    Log message at warning level
    :param message: The message
    :param newline: If to have a newline at the end of the message
    """
    log_message(message, LogLevel.WARNING, newline)


def critical(message: str, newline: bool = True):
    """
    Log message at critical level
    :param message: The message
    :param newline: If to have a newline at the end of the message
    """
    log_message(message, LogLevel.CRITICAL, newline)


def neural_network(agent: str, message):
    """
    Log message at neural network level
    :param agent: The agent name
    :param message: The message
    """
    log_message(f'{agent}: {str(message)}', LogLevel.NEURAL_NETWORK)
