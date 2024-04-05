import random
import chess
import torch
from src.concepts.eval import value_tensor

def minimax_engine(board: chess.Board, depth, alpha, beta, maximizing_player, local_model):
    if board.is_game_over():
        outcome = board.outcome().winner
        if outcome is None:
            return None, torch.as_tensor([.5])
        else:
            return None, torch.as_tensor([1]) if outcome else torch.as_tensor([0])
    
    if depth == 0:
        value = value_tensor(board)
        return None, local_model(value)

    if maximizing_player:
        value = torch.as_tensor([0])
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_engine(board, depth-1, alpha, beta, False, local_model)
            if val > value:
                best_mv = move
                value = val
            board.pop()
            if value > beta:
                break
            alpha = max(alpha, value)
        return best_mv, torch.clamp(value + ((random.random() * 2 - 1) / 100), min=0, max=1)
    else:
        value = torch.as_tensor([1])
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_engine(board, depth-1, alpha, beta, True, local_model)
            if val < value:
                best_mv = move
                value = val
            board.pop()
            if value < alpha:
                break
            beta = min(beta, value)
        return best_mv, torch.clamp(value + ((random.random() * 2 - 1) / 100), min=0, max=1)