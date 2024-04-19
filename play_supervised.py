import random
import chess
import numpy as np
import torch
import torch.nn as nn
from src.minimax_supervised import minimax_supervised

class PredictorFF(nn.Module):

    def __init__(self):
        super(PredictorFF, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = torch.load("supervised_model.pt")

board = chess.Board.from_chess960_pos(random.randint(0, 959))
print(board)
pgn = ""

for mv_idx in range(1000):
    move, val = minimax_supervised(board, 6, torch.tensor([-1000], requires_grad=True, dtype=torch.float64), torch.tensor([1000], requires_grad=True, dtype=torch.float64), board.turn, model)
    print(move)
    if move is None:
        break
    if mv_idx % 2 == 0:
        pgn += f'{mv_idx/2+1}. '
    pgn += f'{board.san(move)} '
    board.push(move)
pgn += board.result()
print(pgn)