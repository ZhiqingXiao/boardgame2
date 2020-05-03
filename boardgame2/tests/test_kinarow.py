import numpy as np
import pytest
import gym

import boardgame2


def test_tictactoe():
    env = gym.make('TicTacToe-v0')
    assert env.observation_space[0].shape == (3, 3)
    assert env.observation_space[1].shape == ()
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.high == [2, 2])
    
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()


def test_gomuku():
    env = gym.make('Gomuku-v0')
    assert env.observation_space[0].shape == (15, 15)
    assert env.observation_space[1].shape == ()
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.high == [14, 14])
    
    observation = env.reset()
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
