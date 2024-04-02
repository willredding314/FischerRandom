import random
import chess
import numpy as np
import torch
from src.material_agent import get_material
from src.predictor import Predictor
import torch.nn as nn
import copy

torch.set_default_dtype(torch.float64)

model = Predictor()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters())

def minimax_modded(board: chess.Board, depth, alpha, beta, maximizing_player, local_model):
    if board.is_game_over():
        outcome = board.outcome().winner
        if outcome is None:
            return None, None
        else:
            return None, torch.as_tensor([1]) if outcome else torch.as_tensor([0])
    
    if depth == 0:
        material_value = get_material(board)
        # THIS GETS REPLACED WITH CONCEPT FUNCTION VALUES
        value_tensor = torch.full([50], material_value, dtype=torch.float64, requires_grad=True)
        return None, local_model(value_tensor)

    if maximizing_player:
        value = torch.as_tensor([0])
        best_mv = None
        for move in board.legal_moves:
            if best_mv is None:
                best_mv = move
            board.push(move)
            mv, val = minimax_modded(board, depth-1, alpha, beta, False, local_model)
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
            mv, val = minimax_modded(board, depth-1, alpha, beta, True, local_model)
            if val < value:
                best_mv = move
                value = val
            board.pop()
            if value < alpha:
                break
            beta = min(beta, value)
        return best_mv, torch.clamp(value + ((random.random() * 2 - 1) / 100), min=0, max=1)

def run_game(board, model):
    values_for_white = []
    values_for_black = []
    white_to_move = True
    
    for i in range(1000):
        move, val = minimax_modded(board, 3, 0, 1, board.turn, model)
        if move is None:
            break
        board.push(move)
        if white_to_move:
            values_for_white.append(board.copy())
        else:
            values_for_black.append(board.copy())
        white_to_move = not white_to_move
    return values_for_white, values_for_black, not white_to_move # return winner (the side that does NOT have the next move)
        
for i in range(1):
    board = chess.Board(chess960=True)
    white_positions, black_positions, white_wins = run_game(board, copy.deepcopy(model))

    white_target = 0
    if white_wins:
        white_target = 1
    
    for position in white_positions:
        optimizer.zero_grad()
        mv, value = minimax_modded(position, 3, 0, 1, True, model)
        if mv is None:
            break
        loss = criterion(value, torch.as_tensor([white_target], dtype=torch.float64))
        loss.backward()
        optimizer.step()

    for position in black_positions:
        optimizer.zero_grad()
        mv, value = minimax_modded(position, 3, 0, 1, False, model)
        if mv is None:
            break
        loss = criterion(value, torch.as_tensor([1 - white_target], dtype=torch.float64))
        loss.backward()
        optimizer.step()

weights = model.fc.weight

print(weights)