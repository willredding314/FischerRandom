import random
import chess

depth = 7


material_map = {'P':1,'B':3,'N':3,'R':5,'Q':9,'K':0,'p':-1,'b':-3,'n':-3,'r':-5,'q':-9,'k':-0}
# Gets the material count for the board
def get_material(board: chess.Board):
    total = 0
    pieces = board.piece_map()
    for square, piece in pieces.items():
        total += material_map[piece.symbol()]
    return total

def minimax_engine_material_only(board: chess.Board, depth, alpha, beta, maximizing_player=True):
    if board.is_game_over():
        outcome = board.outcome().winner
        if outcome is None:
            return None, 0
        else:
            return None, float("inf") if outcome else float("-inf")
    
    if depth == 0:
        return None, get_material(board)

    if maximizing_player:
        value = float("-inf")
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_engine_material_only(board, depth-1, alpha, beta, False)
            if val > value:
                best_mv = move
                value = val
            board.pop()
            if value > beta:
                break
            alpha = max(alpha, value)
        return best_mv, value - random.random()/1e6
    else:
        value = float("inf")
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_engine_material_only(board, depth-1, alpha, beta, True)
            if val < value:
                best_mv = move
                value = val
            board.pop()
            if value < alpha:
                break
            beta = min(beta, value)
        return best_mv, value + random.random()/1e6
    
board = chess.Board()
pgn = ""

for i in range(1000):
    move, val = minimax_engine_material_only(board, depth - 1, float("-inf"), float("inf"), board.turn)
    if move is None:
        break
    if i % 2 == 0:
        pgn += f'{i/2+1}. '
    print(board.san(move))
    pgn += f'{board.san(move)} '
    board.push(move)

pgn += board.result()
print(pgn)