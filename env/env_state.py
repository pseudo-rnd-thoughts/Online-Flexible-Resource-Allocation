"""Immutable Environment state"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from env.task import Task
    from env.server import Server
    from typing import Dict, List, Optional


class EnvState(NamedTuple):
    """The environment state"""

    server_tasks: Dict[Server, List[Task]]
    auction_task: Optional[Task]
    time_step: int

    def __str__(self) -> str:
        server_tasks_str = ', '.join([f'{server.name}: [{", ".join([task.name for task in tasks])}]'
                                      for server, tasks in self.server_tasks.items()])
        auction_task_str = str(self.auction_task) if self.auction_task else 'None'
        return f'Env State ({hex(id(self))}) at time step: {self.time_step}\n' \
               f'Auction Task -> {auction_task_str}\n' \
               f'Servers -> {server_tasks_str}'

    def _repr_pretty_(self, p, cycle):
        p.text(self.__str__())
