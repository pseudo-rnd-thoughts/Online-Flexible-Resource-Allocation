"""The agent class acts as the server class managing resources and bidding in auctions"""

from typing import List, Dict
from tensorflow import keras

from task import Task, TaskStage


class Agent:

    tasks: List[Task] = []

    def __init__(self, storage_capacity: float, computation_capacity: float, bandwidth_capacity: float):
        self.storage_capacity: float = storage_capacity
        self.computation_capacity: float = computation_capacity
        self.bandwidth_capacity: float = bandwidth_capacity

        self.task_pricing_network = self.init_task_pricing_network()
        self.resource_allocation_network = self.init_resource_allocation_network()

    def init_task_pricing_network(self) -> keras.Model:
        inputs = keras.Input(shape=(10, ))

        outputs = keras.layers.Dense(100, activation='relu')
        return keras.Model(inputs, outputs)

    def init_resource_allocation_network(self) -> keras.Model:
        inputs = keras.Input(shape=(10, ))

        outputs = keras.layers.Dense(100, activiation='relu')
        return keras.Model(inputs, outputs)

    def allocate_resources(self):
        """
        In order to allocate resources, the resource allocation neural network is used to calculate
          a weighting (or importance) of each resource.

        Todo: Update to check for adding loading bandwidth when there is no storage for it
        Todo: Update to rebalance the weighting when required
        """

        loading_weights: Dict[Task, int] = {}
        compute_weights: Dict[Task, int] = {}
        sending_weights: Dict[Task, int] = {}

        for task in self.tasks:
            weighting = self.resource_allocation_network(task)

            if task.stage == TaskStage.LOADING:
                loading_weights[task] = weighting
            elif task.stage == TaskStage.COMPUTING:
                compute_weights[task] = weighting
            elif task.stage == TaskStage.SENDING:
                sending_weights[task] = weighting

        bandwidth_unit = self.bandwidth_capacity / (sum(loading_weights.values()) + sum(sending_weights.values()))
        compute_unit = self.computation_capacity / sum(compute_weights.values())

        for task in self.tasks:
            if task.stage == TaskStage.LOADING:
                task.allocate_loading_resources(bandwidth_unit)
            elif task.stage == TaskStage.COMPUTING:
                task.allocate_compute_resources(compute_unit)
            elif task.stage == TaskStage.SENDING:
                task.allocate_sending_resources(bandwidth_unit)
