import chess
import torch
from src.material_agent import minimax
from src.predictor import Predictor
import torch.nn as nn

model = Predictor()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters())

def run_game(board):
    values_for_white = []
    values_for_black = []
    white_to_move = True
    
    for i in range(1000):
        move, val = minimax(board, 3, float("-inf"), float("inf"), board.turn)
        if move is None:
            break
        board.push(move)
        if white_to_move:
            values_for_white.append(val)
        else:
            values_for_black.append(val)
        white_to_move = not white_to_move
    return values_for_white, values_for_black, not white_to_move # return winner (the side that does NOT have the next move)
        
for i in range(1):
    board = chess.Board(chess960=True)
    white_values, black_values, white_wins = run_game(board)
    
    white_target = 0
    if white_wins:
        white_target = 1
    
    for value in white_values:
        optimizer.zero_grad()
        loss = criterion(torch.Tensor(value), torch.Tensor(white_target))
        loss.backward()
        optimizer.step()

    for value in black_values:
        optimizer.zero_grad()
        loss = criterion(torch.Tensor(value), torch.Tensor(1 - white_target))
        loss.backward()
        optimizer.step()

weights = model.fc.weight

print(weights)