"""
Tests the linear exploration method used in the DQN agent
"""

# TODO add comments
from agents.rl_agents.agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import create_lstm_dqn_network
from env.environment import OnlineFlexibleResourceAllocationEnv


def test_epsilon_policy():
    print()
    env = OnlineFlexibleResourceAllocationEnv('agent/settings/basic.env')
    state = env.reset()

    num_servers = len(state.server_tasks.keys())
    pricing_dqn_agent = TaskPricingDqnAgent(0, create_lstm_dqn_network(9, 5), initial_epsilon=1, final_epsilon=0.1,
                                            update_frequency=1,
                                            epsilon_steps=(len(env._unallocated_tasks) + (
                                                1 if state.auction_task is not None else 0)) * num_servers)
    weighting_dqn_agent = ResourceWeightingDqnAgent(0, create_lstm_dqn_network(16, 5), initial_epsilon=1,
                                                    final_epsilon=0.1, update_frequency=1,
                                                    epsilon_steps=env._total_time_steps * num_servers)

    done = False
    while not done:
        if state.auction_task is not None:
            actions = {
                server: pricing_dqn_agent.bid(state.auction_task, tasks, server, state.time_step, training=True)
                for server, tasks in state.server_tasks.items()
            }
        else:
            actions = {
                server: weighting_dqn_agent.weight(tasks, server, state.time_step, training=True)
                for server, tasks in state.server_tasks.items()
            }
        state, rewards, done, _ = env.step(actions)

    assert pricing_dqn_agent.total_actions == pricing_dqn_agent.epsilon_steps
    assert round(pricing_dqn_agent.epsilon, 2) == 0.1

    assert 0 < weighting_dqn_agent.total_actions
