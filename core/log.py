"""Logging of a message to either the console or file"""

from __future__ import annotations

from enum import Enum


class DebugLevel(Enum):
    """Debug level enum"""
    DEBUG = 1
    INFO = 2
    WARNING = 3
    CRITICAL = 4


debug_filename: str = "debug.log"
file_debug_level: DebugLevel = DebugLevel.DEBUG
console_debug_level: DebugLevel = DebugLevel.INFO


def log_message(message: str, debug_level: DebugLevel, newline: bool = True):
    """
    Debug the message to either console or files
    :param message: The string message
    :param debug_level: The debug level
    :param newline: If to append a new line to end of the message
    """
    if debug_level.value <= file_debug_level.value:
        with open(debug_filename, 'w') as file:
            if newline:
                file.write(message + '\n')
            else:
                file.write(message)

    if debug_level.value <= console_debug_level.value:
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
