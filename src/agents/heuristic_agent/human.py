"""
A human agent such that for each action is decided by user input
"""

from agents.task_pricing_agents import TaskPricingAgent


class HumanTaskPricing(TaskPricingAgent):
    """
    Human task pricing agent where a user input is required for each bid
    """

    def _get_action(self, task_states) -> float:
        print(task_states)

        price = -1
        while price == -1:
            try:
                price = int(input('Enter task bid: '))
                if 0 < price:
                    print('Please enter a positive number or zero if not bidding')
                    price = -1
            except ValueError:
                print('Please enter a number')

        return price
