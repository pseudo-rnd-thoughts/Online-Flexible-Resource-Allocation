"""
Deep Deterministic Policy Gradient networks (actor and critic)
"""

from agents.rl_agents.neural_networks.network import Network


class DdpgActor(Network):
    """
    DDPG Actor
    """

    def __init__(self, input_width, max_action_value):
        Network.__init__(self, 'DDPG Actor', input_width, max_action_value)

    def call(self, inputs, training=None, mask=None):
        # todo
        pass


class DdpgCritic(Network):
    """
    DDPG Critic
    """

    def __init__(self, input_width, max_action_value):
        Network.__init__(self, 'DDPG Critic', input_width, max_action_value)

    def call(self, inputs, training=None, mask=None):
        # todo
        pass
