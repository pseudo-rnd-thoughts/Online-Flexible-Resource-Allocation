"""
Tests the linear exploration method used in the DQN agent
"""

# TODO add comments
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from agents.rl_agents.rl_policies import EpsilonGreedyTaskPricingPolicy, EpsilonGreedyResourceAllocationPolicy
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_epsilon_policy():
    print()
    env = OnlineFlexibleResourceAllocationEnv('agent/settings/basic.env')
    state = env.reset()

    pricing_dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5))
    weighting_dqn_agent = ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 5))

    num_servers = len(state.server_tasks.keys())
    pricing_policy = EpsilonGreedyTaskPricingPolicy(
        pricing_dqn_agent, initial_epsilon=1, final_epsilon=0.1, update_frequency=1,
        epsilon_steps=(len(env._unallocated_tasks) + (1 if state.auction_task is not None else 0)) * num_servers)
    weighting_policy = EpsilonGreedyResourceAllocationPolicy(
        weighting_dqn_agent, initial_epsilon=1, final_epsilon=0.1, update_frequency=1,
        epsilon_steps=env._total_time_steps * num_servers)

    done = False
    while not done:
        if state.auction_task is not None:
            actions = {
                server: pricing_policy.bid(state.auction_task, tasks, server, state.time_step)
                for server, tasks in state.server_tasks.items()
            }
        else:
            actions = {
                server: weighting_policy.weight(tasks, server, state.time_step)
                for server, tasks in state.server_tasks.items()
            }
        state, rewards, done, _ = env.step(actions)

    assert pricing_policy.total_actions == pricing_policy.epsilon_steps
    assert round(pricing_policy.epsilon, 2) == 0.1
