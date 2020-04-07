import random as rnd


class RandomAgent:

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def action(self):
        return rnd.randint(0, self.num_actions - 1)
