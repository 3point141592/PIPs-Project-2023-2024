"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy
from pprint import pp

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    numX = sum([row.count(X) for row in board])
    numO = sum([row.count(O) for row in board])
    return X if numX == numO else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    possible = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible.add((i, j))

    return possible


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]]:
        raise Exception

    elif not 0 <= action[0] <= 2:
        raise Exception

    elif not 0 <= action[1] <= 2:
        raise Exception

    
    board = deepcopy(board)
    board[action[0]][action[1]] = player(board)

    return board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    wins = [((0, 0), (0, 1), (0, 2)),
            ((0, 2), (1, 1), (2, 0)),
            ((2, 0), (2, 1), (2, 2)),
            ((0, 1), (1, 1), (2, 1)),
            ((0, 0), (1, 1), (2, 2)),
            ((1, 0), (1, 1), (1, 2)),
            ((0, 0), (1, 0), (2, 0)),
            ((0, 2), (1, 2), (2, 2))]

    for win in wins:
        controlled = [board[cell[0]][cell[1]] for cell in win]
        if controlled[0] != EMPTY and all([value == controlled[0] for value in controlled]):
            return controlled[0]
        
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return True if winner(board) or len([board[x][y] for x in range(len(board)) for y in range(len(board[0])) if board[x][y] == EMPTY]) == 0 else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    
    winning_player = winner(board)
    scores = {X: 1, O: -1}
    return scores.get(winning_player, 0)

def maxvalue(board):
    """
    Returns the Max-value of "board". "func" should be "min" for the min player, "max" for the max player
    """

    if terminal(board):
        return utility(board)

    v = -float('inf')

    for action in actions(board):
        v = max(v, minvalue(result(board, action)))

    return v

def minvalue(board):
    """
    Returns the Min-value of "board". "func" should be "min" for the min player, "max" for the max player
    """

    if terminal(board):
        return utility(board)

    v = float('inf')

    for action in actions(board):
        v = min(v, maxvalue(result(board, action)))

    return v

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # precomputed for time efficiency
    if len(actions(board)) == 9:
        return (0, 1)
    
    if terminal(board):
        return None
    possible = list(actions(board))
    if len(possible) == 1:     # if we only have one option, we're forced to play that option
        return possible[0]
    
    min_or_max = 'max' if player(board) == X else 'min'

    if min_or_max == 'max':
        function = maxvalue
        boolean_func = lambda x, y: x > y

    else:
        function = minvalue
        boolean_func = lambda x, y: x < y
    

    best_move = None
    best_value = None
    depth = 3
    
    for move in possible:
        new = result(board, move) 
        util = function(new)
        if best_move is None:
            best_move = move
            best_value = util

        elif boolean_func(util, best_value):
            best_move = move
            best_value = util
        
    
    
    return best_move
