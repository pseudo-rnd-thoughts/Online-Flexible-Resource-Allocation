"""
Training of single agents with multiple environments
"""

from __future__ import annotations

from agents.rl_agents.agents.dqn import ResourceWeightingDqnAgent, TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from training.train_agents import setup_tensorboard, multi_env_single_env_training

if __name__ == "__main__":
    folder = 'single_agent_multi_envs'
    primary_writer, datetime = setup_tensorboard('training/results/logs/', folder)

    save_folder = f'{folder}_{datetime}'

    task_pricing_agents = [
        TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 21), save_folder=save_folder)
    ]
    resource_weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 11), save_folder=save_folder)
    ]

    multi_env_single_env_training(folder, datetime, primary_writer, task_pricing_agents, resource_weighting_agents)
