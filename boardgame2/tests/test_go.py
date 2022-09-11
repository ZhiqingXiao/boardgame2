import numpy as np
import pytest
import gym

import boardgame2


def test_go():
    env = gym.make('Go-v0', new_step_api=True)
    assert env.observation_space[0].shape == (19, 19)
    assert env.observation_space[1].shape == ()
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.high == [18, 18])

    observation, info = env.reset()
    while False:  # We do not test it, since it has not been fully implemented.
        action = env.action_space.sample()
        observation, reward, termination, truncation, info = env.step(action)
        if termination or truncation:
            break
    env.close()
