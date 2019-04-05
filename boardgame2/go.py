import collections

import numpy as np

from .env import EMPTY, BLACK, WHITE
from .env import BoardGameEnv
from .env import is_index


class GoJudger:
    
    
    def __init__(self, komi):
        self.komi = komi
    
    
    def __call__(self, board):
        self.board = board
        
        self.remove_dead()
        
        # floodfill for both player
        self.fill = np.zeros(board.shape + (3,), dtype=int)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                self.floodfill((x, y), board[x, y])
        
        count_black = np.nansum(self.fill[:, :, BLACK] > self.fill[:, :, WHITE])
        count_white = np.nansum(self.fill[:, :, WHITE] > self.fill[:, :, BLACK])
        if count_black > count_white + self.komi:
            return BLACK
        return WHITE
    
    
    def remove_dead(self):
        pass # TODO
    
    
    def floodfill(self, location, player):
        x, y = location
        if not self.fill[x, y, player]:
            self.fill[x, y, player] = 1
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                xx, yy = x + dx, y + dy
                if is_index(self.board, (xx, yy)) and self.board[xx, yy] != -player:
                    self.floodfill((xx, yy), player)



class GoEnv(BoardGameEnv):
    def __init__(self, board_shape=19, komi=0, allow_suicide=False,
            illegal_action_mode='pass', render_characters='+ox'):
        super().__init__(board_shape=board_shape,
            illegal_action_mode=illegal_action_mode,
            render_characters=render_characters)
        self.judger = GoJudger(komi)
        self.allow_suicide = allow_suicide
        ko_space = spaces.Box(low=0, high=1, shape=observation_space.spaces[0].shape, dtype=np.int8)
        pass_space = spaces.Discrete(2)
        self.observation_space = spaces.Tuple(observation_space.spaces + [ko_space, pass_space])
    
    
    def reset(self):
        super().set_board()
        self.board = np.zeros_like(self.board, dtype=np.int8)
        self.player = BLACK
        self.ko = np.zeros_like(self.board, dtype=np.int8)
        self.pas = False # record pass
        return self.board, self.player, self.ko, self.pas
    
    
    def is_valid(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int, np.array, int)    board, player, ko, pass
        action : np.array   location
        
        Returns
        ----
        valid : book
        """
        board, _, ko, _ = state
        
        if is_index(board, action):
            return False
        
        x, y = action
        
        if board[x, y] or ko[x, y]:
            return False
        
        if not allow_suicide:
            board[x, y] = player # place
            
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                xx, yy = x + dx, y + dy
                if is_index(board, (xx, yy)):
                    if board[xx, yy] == -player:
                        _, liberties = self.search(board, (xx, yy), max_liberty=1)
                        if not liberties:
                            return True
            
            _, my_liberties = self.search(board, (x, y), max_liberty=1)
            if not my_liberties:
                return False
        
        return True
    
    
    def get_winner(self, state):
        """
        Parameters
        ----
        state : (np.array, int, np.array, int)    board, player, ko, pass
        
        Returns
        ----
        winner : None or int
            - None   if the game is not ended and the winner is not determined
            - int    the winner
        """
        raise NotImplementedError()
    
    
    def search(self, board, location, max_liberty=float('+inf'), max_stone=float('+inf')):
        # BFS
        x0, y0 = location
        q = collections.deque()
        q.append((x0, y0))
        locations = set([(x0, y0),])
        liberties = set()
        while q:
            x, y = q.popleft()
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                xx, yy = x + dx, y + dy
                if is_index(board, (xx, yy)):
                    if board[xx, yy] == board[x0, y0]:
                        if (xx, yy) not in locations:
                            q.append((xx, yy))
                            locations.add((xx, yy))
                            if len(locations) >= max_stone:
                                return locations, liberties
                    elif board[xx, yy] == EMPTY:
                        liberties.add((xx, yy))
                        if len(liberties) >= max_liberty:
                            return locations, liberties
        return locations, liberties
    
    
    def get_next_state(self, state, action):
        """
        Parameters
        ----
        state : (np.array, int, np.array, int)        board, player, ko, pass
        action : np.array    location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """
        board, player, _, _ = state
        location = action
        
        ko, pas = np.zeros_like(board), True
        if self.is_valid(state, action):
            pas = False
            
            board[x, y] = player # place
            
            suicides, my_liberties = self.search(board, (x, y), max_liberty=1)
            
            delete_count = 0
            for dx, dy in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                xx, yy = x + dx, y + dy
                if is_index(board, (xx, yy)):
                    if board[xx, yy] == player:
                        deletes, liberties = self.search(board, (xx, yy), max_liberty=1)
                        if not liberties:
                            delete_count += len(locations)
                            for x_del, y_del in locations:
                                board[x_del, y_del] = 0
                            if delete_count == 1:
                                ko[x_del, y_del] = 1
            
            if not my_liberties:
                if delete_count != 1 and len(suicides) != 1:
                    ko = np.zeros_like(board)
                    
                if self.allow_suicide:
                    for x_del, y_del in suicides:
                        board[x_del, y_del] = 0
            
        return board, -player, ko, pas
    
    
    def step(self, action):
        """
        Parameters
        ----
        action : np.array    location
        
        Returns
        ----
        next_state : (np.array, int, np.array, bool)    next board and next player
        reward : float               the winner or zeros
        done : bool           whether the game end or not
        info : {}
        """
        x, y = action
        if not self.valid[x, y]:
            action = self.illegal_equivalent_action
        
        if np.array_equal(action, self.RESIGN):
            self.player = -self.player
            next_state = self.board, self.player, np.zeros_like(self.board), False
            return next_state, self.player, True, {}
        
        self.board, self.player, self.ko, self.pas = self.get_next_state(
                (self.board, self.player, self.ko, self.pas), action)
        while True:
            winner = self.get_winner(self.board)
            if winner is not None:
                return (self.board, self.player, self.ko, self.pas), winner, True, {}
            if self.has_valid((self.board, self.player)):
                break
            self.board, self.player, self.ko, self.pas = self.get_next_state(
                    (self.board, self.player, self.ko, self.pas), self.PASS)
        return (self.board, self.player, self.ko, self.pas), 0., False, {}
