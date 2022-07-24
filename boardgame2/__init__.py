from gym.envs.registration import register
from .env import *
from .reversi import *
from .kinarow import *
from .go import *


register(
        id='Reversi-v0',
        entry_point='boardgame2:ReversiEnv',
        )


register(
        id='KInARow-v0',
        entry_point='boardgame2:KInARowEnv',
        )


register(
        id='Gomuku-v0',
        entry_point='boardgame2:KInARowEnv',
        kwargs={
            'board_shape': 15,
            'target_length': 5,
        }
        )


register(
        id='TicTacToe-v0',
        entry_point='boardgame2:KInARowEnv',
        kwargs={
            'board_shape': 3,
            'target_length': 3,
        }
        )


register(
        id='Go-v0',
        entry_point='boardgame2:GoEnv',
        )
