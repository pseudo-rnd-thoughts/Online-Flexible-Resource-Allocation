"""
Tests the dqn agents training method
"""

from typing import List

from agents.rl_agents.agents.ddpg import TaskPricingDdpgAgent, ResourceWeightingDdpgAgent, ResourceWeightingTD3Agent, \
    TaskPricingTD3Agent
from agents.rl_agents.agents.dqn import ResourceWeightingDqnAgent, ResourceWeightingDdqnAgent, \
    ResourceWeightingDuelingDqnAgent, TaskPricingDqnAgent, TaskPricingDdqnAgent, TaskPricingDuelingDqnAgent, \
    TaskPricingCategoricalDqnAgent, ResourceWeightingCategoricalDqnAgent
from agents.rl_agents.neural_networks.ddpg_networks import create_lstm_actor_network, create_lstm_critic_network
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dueling_dqn_network, create_lstm_dqn_network, \
    create_lstm_categorical_dqn_network
from agents.rl_agents.rl_agents import ResourceWeightingRLAgent, TaskPricingState, TaskPricingRLAgent, \
    ResourceAllocationState
from env.environment import OnlineFlexibleResourceAllocationEnv
from env.task_stage import TaskStage
from training.train_agents import setup_tensorboard


def test_task_price_training():
    print()
    setup_tensorboard('training/tmp/results/', 'price_training')

    # List of agents
    agents: List[TaskPricingRLAgent] = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 10), batch_size=4, save_folder='tmp'),
        TaskPricingDdqnAgent(1, create_lstm_dqn_network(9, 10), batch_size=4, save_folder='tmp'),
        TaskPricingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(9, 10), batch_size=4, save_folder='tmp'),
        TaskPricingCategoricalDqnAgent(3, create_lstm_categorical_dqn_network(9, 10), batch_size=4, save_folder='tmp'),
        TaskPricingDdpgAgent(4, create_lstm_actor_network(9), create_lstm_critic_network(9), batch_size=4,
                             save_folder='tmp'),
        TaskPricingTD3Agent(5, create_lstm_actor_network(9), create_lstm_critic_network(9),
                            create_lstm_critic_network(9), batch_size=4, save_folder='tmp')
    ]

    # Load the environment
    env, state = OnlineFlexibleResourceAllocationEnv.load_env('training/settings/auction.env')

    # Servers
    server_1, server_2 = list(state.server_tasks.keys())
    # Actions
    actions = {server_1: 1.0, server_2: 2.0}

    # Environment step
    next_state, reward, done, info = env.step(actions)

    # Server states
    server_1_state = TaskPricingState(state.auction_task, state.server_tasks[server_1], server_1, state.time_step)
    server_2_state = TaskPricingState(state.auction_task, state.server_tasks[server_2], server_2, state.time_step)

    # Next server states
    next_server_1_state = TaskPricingState(next_state.auction_task, next_state.server_tasks[server_1],
                                           server_1, next_state.time_step)
    next_server_2_state = TaskPricingState(next_state.auction_task, next_state.server_tasks[server_2],
                                           server_2, next_state.time_step)
    # Finished auction task
    finished_task = next(finished_task for finished_task in next_state.server_tasks[server_1]
                         if finished_task == state.auction_task)
    finished_task = finished_task._replace(stage=TaskStage.COMPLETED)
    failed_task = finished_task._replace(stage=TaskStage.FAILED)

    # Loop over the agents, add the observations and try training
    for agent in agents:
        agent.winning_auction_bid(server_1_state, actions[server_1], finished_task, next_server_1_state)
        agent.winning_auction_bid(server_1_state, actions[server_1], failed_task, next_server_1_state)
        agent.failed_auction_bid(server_2_state, actions[server_2], next_server_2_state)
        agent.failed_auction_bid(server_2_state, 0, next_server_2_state)

        agent.train()

    print(f'Rewards: {[trajectory[3] for trajectory in agents[0].replay_buffer]}')


def test_resource_allocation_training():
    print()
    setup_tensorboard('training/tmp/results/', 'resource_allocation_training')

    # List of agents
    agents: List[ResourceWeightingRLAgent] = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 10), batch_size=4, save_folder='tmp'),
        ResourceWeightingDdqnAgent(1, create_lstm_dqn_network(16, 10), batch_size=4, save_folder='tmp'),
        ResourceWeightingDuelingDqnAgent(2, create_lstm_dueling_dqn_network(16, 10), batch_size=4, save_folder='tmp'),
        ResourceWeightingCategoricalDqnAgent(3, create_lstm_categorical_dqn_network(16, 10), batch_size=2,
                                             save_folder='tmp'),
        ResourceWeightingDdpgAgent(4, create_lstm_actor_network(16), create_lstm_critic_network(16), batch_size=4,
                                   save_folder='tmp'),
        ResourceWeightingTD3Agent(5, create_lstm_actor_network(16), create_lstm_critic_network(16),
                                  create_lstm_critic_network(16), batch_size=4, save_folder='tmp')
    ]

    # Load the environment
    env, state = OnlineFlexibleResourceAllocationEnv.load_env('training/settings/resource_allocation.env')

    # Servers and tasks
    server = list(state.server_tasks.keys())[0]
    task_1, task_2, task_3, task_4 = list(state.server_tasks[server])

    # Actions
    actions = {
        server: {
            task_1: 1.0,
            task_2: 3.0,
            task_3: 0.0,
            task_4: 5.0
        }
    }

    # Environment step
    next_state, rewards, done, _ = env.step(actions)

    # Resource state
    resource_state = ResourceAllocationState(state.server_tasks[server], server, state.time_step)
    # Next server and resource state
    next_resource_state = ResourceAllocationState(next_state.server_tasks[server], server, next_state.time_step)

    for agent in agents:
        agent.resource_allocation_obs(resource_state, actions[server], next_resource_state, rewards[server])

        agent.train()
