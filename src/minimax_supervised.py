import random
import chess
import numpy as np
import torch

material_map = {'P':1,'B':3,'N':3,'R':5,'Q':9,'K':0,'p':-1,'b':-3,'n':-3,'r':-5,'q':-9,'k':-0}
def position_tensor(board):
    pieces = board.piece_map()
    value_vector = np.zeros(64)
    for square, piece in pieces.items():
        value_vector[square] = material_map[piece.symbol()]
    return torch.tensor(value_vector, dtype=torch.float64, requires_grad=True)

def minimax_supervised(board: chess.Board, depth, alpha, beta, maximizing_player, local_model):
    if board.is_game_over():
        outcome = board.outcome().winner
        if outcome is None:
            return None, torch.tensor([0], requires_grad=True, dtype=torch.float64)
        else:
            return None, torch.tensor([1000], requires_grad=True, dtype=torch.float64) if outcome else torch.tensor([-1000], requires_grad=True, dtype=torch.float64)
    
    if depth == 0:
        tensor = position_tensor(board)
        return None, local_model(tensor)

    if maximizing_player:
        value = torch.tensor([-1000], requires_grad=True, dtype=torch.float64)
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_supervised(board, depth-1, alpha, beta, False, local_model)
            if val > value:
                best_mv = move
                value = val
            board.pop()
            if value > beta:
                break
            alpha = max(alpha, value)
        return best_mv, value + (random.random() * 10 - 5)
    else:
        value = torch.tensor([1000], requires_grad=True, dtype=torch.float64)
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_supervised(board, depth-1, alpha, beta, True, local_model)
            if val < value:
                best_mv = move
                value = val
            board.pop()
            if value < alpha:
                break
            beta = min(beta, value)
        return best_mv, value + (random.random() * 10 - 5)