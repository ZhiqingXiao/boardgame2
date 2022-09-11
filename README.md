boardgame2
=======================

`boardgame2` is an extension of OpenAI Gym that implements multiple two-player zero-sum 2-dimension board games, such as TicTacToe, Gomuko, and Reversi.


## Environments
- `Reversi-v0`
- `KInARow-v0`, as well as `Gomuku-v0` and `TicTacToe-v0`
- `Go-v0` (Experimental, not fully implemented)

## Install

    pip install --upgrade boardgame2

We support Windows, macOS, Linux, and other operating systems.


## Usage

See [API docs](http://github.com/zhiqingxiao/boardgame2/blob/master/doc/api.md) for all classes and functions.


Create a Game

```
import gym
import boardgame2

env = gym.make('TicTacToe-v0') # 3x3, 3-in-a-row
env = gym.make('Gomuku-v0') # 15x15, 5-in-a-row
env = gym.make('KInARow-v0', board_shape=5, target_length=4) # 5x5, 4-in-a-row
env = gym.make('KInARow-v0', board_shape=(3, 5), target_length=4) # 3x5, 4-in-a-row
env = gym.make('Reversi-v0') # 8x8
env = gym.make('Reversi-v0', board_shape=6) # 6x6
env = gym.make('Go-v0') # 19x19
env = gym.make('Go-v0', board_shape=15) # 15x15
```

Play a Game

```
import gym
import boardgame2

env = gym.make('TicTacToe-v0')
print('observation space = {}'.format(env.observation_space))
print('action space = {}'.format(env.action_space))

observation, info = env.reset()
while True:
    action = env.action_space.sample()
    observation, reward, termination, truncation, info = env.step(action)
    if termination or truncation:
        break
env.close()
```

# BibTeX

This package has been published in the following book:

    @book{xiao2019,
     title     = {Reinforcement Learning: Theory and {Python} Implementation},
     author    = {Zhiqing Xiao}
     year      = 2019,
     month     = 8,
     publisher = {China Machine Press},
    }
