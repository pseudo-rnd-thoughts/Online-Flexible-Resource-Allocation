
def test_linear_exploration():
    initial_exploration = 1
    final_exploration = 0.1
    final_exploration_frame = 100000

    for obs in range(0, final_exploration_frame, 1000):
        update_exploration = obs * (final_exploration - initial_exploration) / final_exploration_frame + initial_exploration
        print(update_exploration)


test_linear_exploration()
