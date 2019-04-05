boardgame2 API Complete List
=======================

## Constants

**boardgame2.BLACK**

The 1st player.

**boardgame2.WHITE**

The 2nd player.

**boardgame2.EMPTY**

Neither of the two players.

## Functions

**boardgame2.strfboard**
```
strfboard(board:np.array, render_characters:str='+ox', end:str='\n') -> str
```
Format a board as a string

**boardgame2.is_index**
```
is_index(board:np.array, location:np.array) -> bool
```
Check whether a location is a valid index for the board

**boardgame2.extend_board**
```
extend_board(board:np.array)
```
Get the rotations of the board. Only valid for square board.

## Classes

**boardgame2.BoardGameEnv**

The base class of all board game environment.

```
__init__(board_shape, illegal_action_mode:str='resign', render_characters:str='+ox', allow_pass:bool=True) -> boardgame2.BoardGameEnv
```
Constructor.
board_shape can be either an `int` or `(int, int)`.

```
seed(seed=None) -> NoneType
```
See `gym.Env.seed()`.

```
reset() -> tuple
```
See `gym.Env.reset()`.
observation is in the form of `(np.array, int)`.

```
step(action:np.array) -> tuple, float, bool, dict
```
See `gym.Env.step()`.

```
render(mode:str='human')
```
See `gym.Env.render()`.

```
is_valid(state:tuple, action:np.array) -> bool
```
Check whether the action is valid for current state.

```
get_valid(state:tuple) -> np.array
```
Get all valid locations for the current state.

```
has_valid(state:tuple) -> bool
```
Check whether there are valid locations for current state.

```
get_winner(state:tuple)
```
Check whether the game has ended. If so, who is the winner.

```
get_next_state(state:tuple, action:np.array) -> tuple
```
Get the next state.

```
next_step(state:tuple, action:np.array) -> tuple, float, bool, dict
```
Get the next observation, reward, done, and info. Similar to `gym.Env.step()`.


```
observation_space
```
See `gym.Env.observation_space`.

```
action_space
```
See `gym.Env.action_space`.


```
PASS
```
The action 'skip' (constant).

```
RESIGN
```
The action 'resign' (constant).


**boardgame2.KInARowEnv** (registered as `KInARow-v0`, as well as `Gomuku-v0` and `TicTacToe-v0`)
```
__init__(board_shape, target_length:int=3, illegal_action_mode:str='pass', render_characters:str='+ox') -> boardgame2.KInARowEnv
```


**boardgame2.ReversiEnv** (registered as `Reversi-v0`)
```
__init__(board_shape, render_characters:str='+ox') -> boardgame2.ReversiEnv
```


**boardgame2.GoEnv** (registered as `Go-v0`, not fully implemented)
```
__init__(board_shape, komi:float=0., allow_suicide:bool=False, illegal_action_mode:str='pass', render_characters:str='+ox') -> boardgame2.GoEnv
```

