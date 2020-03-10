"""Tests in agent implementation is valid"""

from agents.rl_agents.ddqn import TaskPricingDdqnAgent, ResourceWeightingDdqnAgent
from agents.rl_agents.dqn import TaskPricingDqnAgent, ResourceWeightingDqnAgent
from agents.rl_agents.dueling_dqn import TaskPricingDuelingDqnAgent, ResourceWeightingDuelingDqnAgent
from agents.rl_agents.neural_networks.dqn_networks import DqnLstmNetwork, DqnBidirectionalLstmNetwork, DqnGruNetwork
from agents.rl_agents.neural_networks.dueling_dqn_networks import DuelingDqnLstmNetwork

if __name__ == '__main__':
    bidirectional_lstm = DqnBidirectionalLstmNetwork(10, 10)
    lstm = DqnLstmNetwork(10, 10)
    gru = DqnGruNetwork(10, 10)
    dueling_lstm = DuelingDqnLstmNetwork(10, 10)
    print(f'{bidirectional_lstm.network_name}, {lstm.network_name}, {gru.network_name}, {dueling_lstm.network_name}')

    tp_dqn_agent = TaskPricingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))
    tp_ddqn_agent = TaskPricingDdqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))
    tp_dueling_dqn_agent = TaskPricingDuelingDqnAgent(0, DqnBidirectionalLstmNetwork(9, 10))

    rw_dqn_agent = ResourceWeightingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))
    rw_ddqn_agent = ResourceWeightingDdqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))
    rw_dueling_dqn_agent = ResourceWeightingDuelingDqnAgent(0, DqnBidirectionalLstmNetwork(10, 10))

    print(f'{id(tp_dqn_agent.model_network)}, {id(tp_dqn_agent.target_network)}')
    print(f'{tp_dqn_agent.name}, {tp_ddqn_agent.name}, {tp_dueling_dqn_agent.name}')
    print(f'{rw_dqn_agent.name}, {rw_ddqn_agent.name}, {rw_dueling_dqn_agent.name}')
