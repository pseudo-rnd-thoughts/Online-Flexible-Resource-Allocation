from agents.task_pricing_agent import TaskPricingAgent
from env.server import Server
from env.task import Task
from env.task_stage import TaskStage

import tensorflow as tf


def network_differential():
    network = TaskPricingNetwork(num_outputs=10)

    server = Server('Test', 220.0, 35.0, 22.0)
    auction_task = Task('Test 4', 69.0, 35.0, 10.0, 0, 12)
    allocated_tasks = [
        Task('Test 1', 76.0, 36.0, 16.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=76.0),
        Task('Test 2', 75.0, 37.0, 12.0, 0, 12, stage=TaskStage.COMPUTING, loading_progress=75.0, compute_progress=10.0),
        Task('Test 3', 72.0, 47.0, 20.0, 0, 7, stage=TaskStage.COMPUTING, loading_progress=72.0, compute_progress=25.0)
    ]
    optimiser = tf.keras.optimizers.RMSprop(lr=0.001)

    observation = TaskPricingAgent.network_observation(auction_task, allocated_tasks, server, 0)
    network(observation)

    network_variables = network.trainable_variables
    with tf.GradientTape() as tape:
        observation = TaskPricingAgent.network_observation(auction_task, allocated_tasks, server, 0)

        action_q_values = network(observation)
        print(f'Action Q Values: {action_q_values}')

        error = tf.reduce_mean(0.5 * tf.square(action_q_values))

    network_gradients = tape.gradient(error, network_variables)
    optimiser.apply_gradients(zip(network_gradients, network_variables))
    print(f'Error: {error}')
    print(f'Network Gradients: {network_gradients}')


network_differential()
