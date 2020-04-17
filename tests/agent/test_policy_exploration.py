"""
Tests the linear exploration method used in the DQN agent
"""

# TODO add comments


def test_linear_exploration():
    initial_exploration = 1
    final_exploration = 0.1
    final_exploration_frame = 100000
    update_exploration = 0

    for obs in range(0, final_exploration_frame+1, 1000):
        update_exploration = obs * (final_exploration - initial_exploration) / final_exploration_frame + initial_exploration
        print(f'Obs: {obs}, Exploration: {update_exploration}')

    assert round(update_exploration, 5) == final_exploration
