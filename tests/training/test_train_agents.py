"""
Tests the src/training/scripts/train_agents.py file
"""

import os
from typing import List

from agents.heuristic_agents.random_agent import RandomTaskPricingRLAgent, RandomResourceWeightingRLAgent
from agents.rl_agents.agents.ddpg import ResourceWeightingDdpgAgent, TaskPricingDdpgAgent
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.ddpg_networks import create_lstm_actor_network, create_lstm_critic_network
from agents.rl_agents.neural_networks.dqn_networks import create_bidirectional_dqn_network, create_rnn_dqn_network
from agents.rl_agents.rl_agents import ReinforcementLearningAgent
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.train_agents import generate_eval_envs, eval_agent, train_agent, setup_tensorboard


def test_agent_evaluation():
    print()
    setup_tensorboard('training/results/tmp/', 'agent_eval')

    env = OnlineFlexibleResourceAllocationEnv('training/settings/basic.env')

    eval_envs = generate_eval_envs(env, 5, 'training/settings/tmp/', overwrite=True)
    assert len(os.listdir('training/settings/tmp/')) == 5
    total_auctions, total_resource_allocation = 0, 0
    for eval_env in eval_envs:
        env, state = OnlineFlexibleResourceAllocationEnv.load_env(eval_env)
        total_auctions += len(env._unallocated_tasks) + (1 if state.auction_task is not None else 0)
        total_resource_allocation += env._total_time_steps + 1

    pricing_agents = [
        TaskPricingDqnAgent(0, create_bidirectional_dqn_network(9, 5)),
        TaskPricingDdpgAgent(1, create_lstm_actor_network(9), create_lstm_critic_network(9))
    ]
    weighting_agents = [
        ResourceWeightingDqnAgent(2, create_bidirectional_dqn_network(16, 5)),
        ResourceWeightingDdpgAgent(3, create_lstm_actor_network(16), create_lstm_critic_network(16)),
    ]

    results = eval_agent(eval_envs, 0, pricing_agents, weighting_agents)
    print(f'Results - Total prices: {results.total_prices}, Number of completed tasks: {results.num_completed_tasks}, '
          f'failed tasks: {results.num_failed_tasks}, winning prices: {results.winning_prices}, '
          f'Number of auctions: {results.num_auctions}, resource allocations: {results.num_resource_allocations}')
    assert 0 < results.num_completed_tasks
    assert 0 < results.num_failed_tasks

    assert results.num_auctions == total_auctions
    assert results.num_resource_allocations == total_resource_allocation


def test_train_agents():
    print()
    setup_tensorboard('training/tmp/', 'train_agents')

    env = OnlineFlexibleResourceAllocationEnv('training/settings/basic.env')

    pricing_agents = [
        TaskPricingDqnAgent(0, create_rnn_dqn_network(9, 5), batch_size=16, initial_training_replay_size=16,
                            training_freq=100),
    ]
    weighting_agents = [
        ResourceWeightingDqnAgent(2, create_rnn_dqn_network(16, 5), batch_size=16,
                                  initial_training_replay_size=16, training_freq=100)
    ]

    for _ in range(2):
        train_agent(env, pricing_agents, weighting_agents)

    # noinspection PyTypeChecker
    agents: List[ReinforcementLearningAgent] = pricing_agents + weighting_agents
    for agent in agents:
        assert 0 < agent.total_updates


def test_train_rnd_agents():
    print()
    setup_tensorboard('training/tmp/', 'train_rnd_agents')
    env = OnlineFlexibleResourceAllocationEnv([
        '../src/training/settings/basic.env',
        '../src/training/settings/large_tasks_servers.env',
        '../src/training/settings/mixture_tasks_servers.env',
        '../src/training/settings/limited_resources.env',
    ])

    pricing_agents = [RandomTaskPricingRLAgent(0)]
    weighting_agents = [RandomResourceWeightingRLAgent(0)]

    for _ in range(10):
        train_agent(env, pricing_agents, weighting_agents)
