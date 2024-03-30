import chess
import random

'''
How to use chess:
Create new board
chess.Board(chess960=True)

Get legal moves (as iterator)
board.legal_moves 

Make moves
board.push(move)
board.pop()
'''
material_map = {'P':1,'B':3,'N':3,'R':5,'Q':9,'K':0,'p':-1,'b':-3,'n':-3,'r':-5,'q':-9,'k':-0}
# Gets the material count for the board
def get_material(board: chess.Board):
    total = 0
    pieces = board.piece_map()
    for square, piece in pieces.items():
        total += material_map[piece.symbol()]
    return total

# Performs minimax on a given depth with alpha-beat pruning
def minimax(board: chess.Board, depth, alpha, beta, maximizing_player=True):
    if board.is_game_over():
        outcome = board.outcome().winner
        if outcome is None:
            return 0
        else:
            return float("inf") if outcome else float("-inf")

    if depth == 0:
        return get_material(board)

    if maximizing_player:
        value = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            value = max(value, minimax(board, depth-1, alpha, beta, False))
            board.pop()
            if value > beta:
                break
            alpha = max(alpha, value)
        return value
    else:
        value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = min(value, minimax(board, depth-1, alpha, beta, True))
            board.pop()
            if value < alpha:
                break
            beta = min(beta, value)
        return value

# Given a board state and a depth, searches that depth on each move.
# This method logic should likely be removed later and moved into the minimax method.
def get_move(board: chess.Board, depth):
    best_value = float("-inf") if board.turn else float("inf")
    best_moves = []
    if board.turn:
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth-1, float("-inf"), float("inf"), False)
            board.pop()
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
    else:
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth-1, float("-inf"), float("inf"), True)
            board.pop()
            if value < best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)
    if len(best_moves) == 0:
        return None, best_value
    return random.choice(best_moves), best_value

# Initiating board
board = chess.Board(chess960=True)
#board.set_chess960_pos(340)

# Goes through the game and creates a pgn and prints it.
print(board)
pgn = ""
for i in range(1000):
    move, val = get_move(board, 3)
    if move is None:
        break
    if i % 2 == 0:
        pgn += f'{i/2+1}. '
    print(board.san(move))
    pgn += f'{board.san(move)} '
    #print(val)
    board.push(move)
    #print(board)

pgn += board.result()
print(pgn)
    




