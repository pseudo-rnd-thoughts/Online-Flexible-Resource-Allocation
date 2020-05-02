"""
Training of multiple agents with a single training environment
"""

from __future__ import annotations

from agents.rl_agents.agents.dqn import ResourceWeightingDqnAgent, TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from training.train_agents import setup_tensorboard, multi_env_single_env_training

if __name__ == "__main__":
    folder = 'multi_agents_single_env'
    primary_writer, datetime = setup_tensorboard('training/results/logs/', folder)

    save_folder = f'{folder}_{datetime}'

    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, create_lstm_dqn_network(9, 21), save_folder=save_folder)
        for agent_num in range(3)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(agent_num, create_lstm_dqn_network(16, 11), save_folder=save_folder)
        for agent_num in range(2)
    ]

    multi_env_single_env_training(folder, datetime, primary_writer, task_pricing_agents, resource_weighting_agents,
                                  multi_env_training=False)
