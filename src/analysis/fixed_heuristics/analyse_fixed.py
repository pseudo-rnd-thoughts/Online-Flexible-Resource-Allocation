"""
Analyses fixed vs flexible resource allocation
"""

from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from analysis.fixed_heuristics.fixed_env import fixed_resource_allocation_model
from env.environment import OnlineFlexibleResourceAllocationEnv
from training.train_agents import eval_agent, generate_eval_envs


def eval_fixed_env(eval_envs_filename):
    total_completed_tasks = []
    for eval_env_filename in eval_envs_filename:
        env, state = OnlineFlexibleResourceAllocationEnv.load_env(eval_env_filename)

        total_completed_tasks.append(fixed_resource_allocation_model(env, state))

    return total_completed_tasks


def load_agents():
    task_pricing_agents = [
        TaskPricingDqnAgent(agent_num, create_lstm_dqn_network(9, 21))
        for agent_num in range(3)
    ]
    task_pricing_agents[0].model_network.load_weights('./analysis/fixed_heuristics/eval_agents/Task_pricing_Dqn_agent_0/update_80922')
    task_pricing_agents[1].model_network.load_weights('./analysis/fixed_heuristics/eval_agents/Task_pricing_Dqn_agent_1/update_86909')
    task_pricing_agents[2].model_network.load_weights('./analysis/fixed_heuristics/eval_agents/Task_pricing_Dqn_agent_2/update_88937')

    resource_weighting_agents = [
        ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 11))
    ]
    resource_weighting_agents[0].model_network.load_weights('./analysis/fixed_heuristics/eval_agents/Resource_weighting_Dqn_agent_0/update_440898')

    return task_pricing_agents, resource_weighting_agents


if __name__ == "__main__":
    eval_env = OnlineFlexibleResourceAllocationEnv([
        './analysis/fixed_heuristics/settings/basic.env',
        './analysis/fixed_heuristics/settings/large_tasks_servers.env',
        './analysis/fixed_heuristics/settings/limited_resources.env',
        './analysis/fixed_heuristics/settings/mixture_tasks_servers.env'
    ])
    eval_envs = generate_eval_envs(eval_env, 12, f'./analysis/fixed_heuristics/eval_envs/')

    task_pricing_agents, resource_weighting_agents = load_agents()
    agent_results = eval_agent(eval_envs, 0, task_pricing_agents, resource_weighting_agents)
    print('Agent results')
    print(f'Env completed tasks: {agent_results.env_completed_tasks}')
    print(f'Env failed tasks: {agent_results.env_failed_tasks}')
    print(f'Env attempted tasks: {agent_results.env_attempted_tasks}\n')

    fixed_results = eval_fixed_env(eval_envs)
    print(f'Fixed results env completed tasks: {fixed_results}')
