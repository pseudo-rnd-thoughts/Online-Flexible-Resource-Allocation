"""Immutable Environment state"""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from env.task import Task
    from env.server import Server


class EnvState(NamedTuple):
    """The environment state"""

    server_tasks: Dict[Server, List[Task]]
    auction_task: Optional[Task] = None

    def __str__(self) -> str:
        server_tasks_str = '\n\t'.join([f'{str(server)}, Tasks: [{", ".join([task.name for task in tasks])}]'
                                        for server, tasks in self.server_tasks.items()])
        auction_task_str = str(self.auction_task) if self.auction_task else 'None'
        return f'Env State ({hex(id(self))})\nAuction Task -> {auction_task_str}\n' \
               f'Servers\n\t{server_tasks_str}\n'

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
