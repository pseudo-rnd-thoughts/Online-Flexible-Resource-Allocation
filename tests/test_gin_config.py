"""
Tests the use of gin configurable with agent and network classes
"""

import gin

from agents.rl_agents.dqn import TaskPricingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork

gin.parse_config("""
import agents.rl_agents.rl_agent

ReinforcementLearningAgent.save_frequency = 4
""")

def test_agent_gin():
    test_agent = TaskPricingDqnAgent(1, DqnLstmNetwork(1, 9, 10))
    assert test_agent.save_frequency == 4
