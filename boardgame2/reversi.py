import copy
import itertools

import numpy as np

from .env import EMPTY
from .env import BoardGameEnv
from .env import is_index


class ReversiEnv(BoardGameEnv):
    
    def __init__(self, board_shape=8, render_characters='+ox'):
        super().__init__(board_shape=board_shape,
            illegal_action_mode='resign', render_characters=render_characters,
            allow_pass=False) # reversi does not allow pass
    
    
    def reset(self):
        super().reset()
        
        x, y = (s // 2 for s in self.board.shape)
        self.board[x - 1][y - 1] = self.board[x][y] = 1
        self.board[x - 1][y] = self.board[x][y - 1] = -1
        
        return self.board, self.player
    
    
    def is_valid(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int)    board and player
        action : np.array   location
        
        Returns
        ----
        valid : np.array     current valid place for the player
        """
        board, player = copy.deepcopy(state)
        
        if not is_index(board, action):
            return False
        
        x, y = action
        if board[x, y] != EMPTY:
            return False
        
        for dx in [-1, 0, 1]: # loop on the 8 directions
            for dy in [-1, 0, 1]:
                if (dx, dy) == (0, 0):
                    continue
                xx, yy = x, y
                for count in itertools.count():
                    xx, yy = xx + dx, yy + dy
                    if not is_index(board, (xx, yy)):
                        break
                    if board[xx, yy] == EMPTY:
                        break
                    if board[xx, yy] == -player:
                        continue
                    if count: # and is player
                        return True
                    break
        return False
    
    
    def get_next_state(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """
        board, player = copy.deepcopy(state)
        if self.is_valid(state, action):
            x, y = action
            board[x, y] = player
            for dx in [-1, 0, 1]: # loop on the 8 directions
                for dy in [-1, 0, 1]:
                    if (dx, dy) == (0, 0):
                        continue
                    xx, yy = x, y
                    for count in itertools.count():
                        xx, yy = xx + dx, yy + dy
                        if not is_index(board, (xx, yy)):
                            break
                        if board[xx, yy] == EMPTY:
                            break
                        if board[xx, yy] == player:
                            for i in range(count+1): # overwrite
                                board[x + i * dx, y + i * dy] = player
                            break
        return board, -player
