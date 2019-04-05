import sys
import copy
import itertools

from six import StringIO
import numpy as np
import gym
from gym import spaces


EMPTY = 0
BLACK = 1
WHITE = -1


def strfboard(board, render_characters='+ox', end='\n'):
    """
    Format a board as a string
    
    Parameters
    ----
    board : np.array
    render_characters : str
    end : str
    
    Returns
    ----
    s : str
    """
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            c = render_characters[board[x][y]]
            s += c
        s += end
    return s[:-len(end)]


def is_index(board, location):
    """
    Check whether a location is a valid index of the board
    
    Parameters:
    ----
    board : np.array
    location : np.array
    
    Returns
    ----
    is_index : bool
    """
    if len(location) != 2:
        return False
    x, y = location
    return x in range(board.shape[0]) and y in range(board.shape[1])


def extend_board(board):
    """
    Get the rotations of the board.
    
    Parameters:
    ----
    board : np.array, shape (n, n)
    
    Returns
    ----
    boards : np.array, shape (8, n, n)
    """
    assert board.shape[0] == board.shape[1]
    boards = np.stack([board,
            np.rot90(board), np.rot90(board, k=2), np.rot90(board, k=3),
            np.transpose(board), np.flipud(board),
            np.rot90(np.flipud(board)), np.fliplr(board)])
    return boards


class BoardGameEnv(gym.Env):
    metadata = {"render.modes": ["ansi", "human"]}
    
    PASS = np.array([-1, 0])
    RESIGN = np.array([-1, -1])
    
    def __init__(self, board_shape,
            illegal_action_mode='resign', render_characters='+ox',
            allow_pass=True):
        """
        Greate a board game.
        
        Parameters
        ----
        board_shape: int or tuple    shape of the board
            - int: the same as (int, int)
            - tuple: in the form of (int, int), the two dimension of the board
        illegal_action_mode: str  What to do when the agent makes an illegal place.
            - 'resign' : invalid location equivalent to resign
            - 'pass' : invalid location equivalent to pass
        render_characters: str with length 3. characters used to render ('012', ' ox', etc)
        """
        self.allow_pass = allow_pass
        
        if illegal_action_mode == 'resign':
            self.illegal_equivalent_action = self.RESIGN
        elif illegal_action_mode == 'pass':
            self.illegal_equivalent_action = self.PASS
        else:
            raise ValueError()
        
        self.render_characters = {player : render_characters[player] for player \
                in [EMPTY, BLACK, WHITE]}
        
        if isinstance(board_shape, int):
            board_shape = (board_shape, board_shape)
        assert len(board_shape) == 2 # invalid board shape
        self.board = np.zeros(board_shape)
        assert self.board.size > 1 # Invalid board shape
        
        observation_spaces = [
                spaces.Box(low=-1, high=1, shape=board_shape, dtype=np.int8),
                spaces.Box(low=-1, high=1, shape=(), dtype=np.int8)]
        self.observation_space = spaces.Tuple(observation_spaces)
        action_spaces = [spaces.Box(low=-np.ones((2,)),
                high=np.array(board_shape)-1, dtype=np.int8),]
        self.action_space = spaces.Tuple(action_spaces)
    
    
    def seed(self, seed=None):
        return []
    
    
    def reset(self):
        """
        Reset a new game episode. See gym.Env.reset()
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        """
        self.board = np.zeros_like(self.board, dtype=np.int8)
        self.player = BLACK
        return self.board, self.player
    
    
    def is_valid(self, state, action):
        """
        Check whether the action is valid for current state.
        
        Parameters
        ----
        state : (np.array, int)    board and player
        action : np.array   location and skip
        
        Returns
        ----
        valid : bool
        """
        board, _ = state
        if not is_index(board, action):
            return False
        x, y = action
        return board[x, y] == EMPTY
    
    
    def get_valid(self, state):
        """
        Get all valid locations for the current state.
        
        Parameters
        ----
        state : (np.array, int)    board and player
        
        Returns
        ----
        valid : np.array     current valid place for the player
        """
        board, _ = state
        valid = np.zeros_like(board, dtype=np.int8)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                valid[x, y] = self.is_valid(state, np.array([x, y]))
        return valid
    
    
    def has_valid(self, state):
        """
        Check whether there are valid locations for current state.
        
        Parameters
        ----
        state : (np.array, int)    board and player
        
        Returns
        ----
        has_valid : bool
        """
        board = state[0]
        valid = np.zeros_like(board, dtype=np.int8)
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if self.is_valid(state, np.array([x, y])):
                    return True
        return False
    
    
    def get_winner(self, state):
        """
        Check whether the game has ended. If so, who is the winner.
        
        Parameters
        ----
        state : (np.array, int)   board and player. only board info is used
        
        Returns
        ----
        winner : None or int
            - None       The game is not ended and the winner is not determined.
            - env.BLACK  The game is ended with the winner BLACK.
            - env.WHITE  The game is ended with the winner WHITE.
            - env.EMPTY  The game is ended tie.
        """
        board, _ = state
        for player in [BLACK, WHITE]:
            if self.has_valid((board, player)):
                return None
        return np.sign(np.nansum(board))
    
    
    def get_next_state(self, state, action):
        """
        Get the next state.
        
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location and skip indicator
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        
        Raise
        ----
        ValueError : location in action is not valid
        """
        board, player = state
        x, y = action
        if self.is_valid(state, action):
            board = copy.deepcopy(board)
            board[x, y] = player
        return board, -player
    
    
    def next_step(self, state, action):
        """
        Get the next observation, reward, done, and info.
        
        Parameters
        ----
        state : (np.array, int)    board and current player
        action : np.array    location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float               the winner or zeros
        done : bool           whether the game end or not
        info : {'valid' : np.array}    a dict shows the valid place for the next player
        """
        if not self.is_valid(state, action):
            action = self.illegal_equivalent_action
        if np.array_equal(action, self.RESIGN):
            return state, -state[1], True, {}
        while True:
            state = self.get_next_state(state, action)
            winner = self.get_winner(state)
            if winner is not None:
                return state, winner, True, {}
            if self.has_valid(state):
                break
            action = self.PASS
        return state, 0., False, {}
    
    
    def step(self, action):
        """
        See gym.Env.step().
        
        Parameters
        ----
        action : np.array    location
        
        Returns
        ----
        next_state : (np.array, int)    next board and next player
        reward : float        the winner or zero
        done : bool           whether the game end or not
        info : {}
        """
        state = (self.board, self.player)
        next_state, reward, done, info = self.next_step(state, action)
        self.board, self.player = next_state
        return next_state, reward, done, info
    
    
    def render(self, mode='human'):
        """
        See gym.Env.render().
        """
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = strfboard(self.board, self.render_characters)
        outfile.write(s)
        if mode != 'human':
            return outfile
