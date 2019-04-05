import itertools

from .env import BLACK, WHITE
from .env import BoardGameEnv
from .env import is_index


class KInARowEnv(BoardGameEnv):
    def __init__(self, board_shape=3, target_length=3,
            illegal_action_mode='pass', render_characters='+ox'):
        super().__init__(board_shape=board_shape,
            illegal_action_mode=illegal_action_mode,
            render_characters=render_characters)
        self.target_length = target_length
    
    
    def get_winner(self, state):
        """
        Parameters
        ----
        state : (np.array, int)   board and player. player info is not used
        
        Returns
        ----
        winner : None or int
            - None   if the game is not ended and the winner is not determined
            - int    the winner
        """
        board, _ = state
        for player in [BLACK, WHITE]:
            for x in range(board.shape[0]):
                for y in range(board.shape[1]):
                    for dx, dy in [(1, -1), (1, 0), (1, 1), (0, 1)]: # loop on the 8 directions
                        xx, yy = x, y
                        for count in itertools.count():
                            if not is_index(board, (xx, yy)) or board[xx, yy] != player:
                                break
                            xx, yy = xx + dx, yy + dy
                        if count >= self.target_length:
                            return player
        for player in [BLACK, WHITE]:
            if self.has_valid((board, player)):
                return None
        return 0
