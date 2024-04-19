import copy
import random
import time
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stockfish import Stockfish
torch.set_default_dtype(torch.float64)
stockfish = Stockfish(path="/usr/local/Cellar/stockfish/16.1/bin/stockfish")
stockfish.update_engine_parameters({"UCI_Chess960": "true"})

material_map = {'P':1,'B':3,'N':3,'R':5,'Q':9,'K':0,'p':-1,'b':-3,'n':-3,'r':-5,'q':-9,'k':-0}
# Gets the material count for the board
def get_material(board: chess.Board):
    total = 0
    pieces = board.piece_map()
    for square, piece in pieces.items():
        total += material_map[piece.symbol()]
    return total

def position_tensor(board):
    pieces = board.piece_map()
    value_vector = np.zeros(64)
    for square, piece in pieces.items():
        value_vector[square] = material_map[piece.symbol()]
    return torch.tensor(value_vector, dtype=torch.float64, requires_grad=True)
       

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
        return best_mv, value + (random.random() * 10 - 1)
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
        return best_mv, value + (random.random() * 10 - 1)
    
def run_game(board, model, depth):
    positions_for_white = []
    positions_for_black = []
    white_to_move = True
    
    while True:
        move, val = minimax_supervised(board, depth - 1, torch.tensor([-1000], requires_grad=True, dtype=torch.float64), torch.tensor([1000], requires_grad=True, dtype=torch.float64), board.turn, model)
        if move is None:
            break
        board.push(move)
        white_to_move = not white_to_move

    return positions_for_white, positions_for_black, not white_to_move

def train_with_random_games(model, criterion, optimizer, num_games, depth):
    for i in range(num_games):

        board = chess.Board.from_chess960_pos(random.randint(0, 959))
        with open('free_out.txt', 'a') as f:
            f.write(board.fen() + "\n")

        pgn = ""
        for mv_idx in range(1000):
            move, val = minimax_supervised(board, depth - 1, torch.tensor([-1000], requires_grad=True, dtype=torch.float64), torch.tensor([1000], requires_grad=True, dtype=torch.float64), board.turn, model)
            print(move)
            if move is None:
                break
            
            if mv_idx % 2 == 0:
                pgn += f'{mv_idx/2+1}. '
            pgn += f'{board.san(move)} '
            
            board.push(move)

            optimizer.zero_grad()
            loss = criterion(val, torch.as_tensor([get_sf_eval(board)], dtype=torch.float64))
            loss.backward()
            optimizer.step()
            
        pgn += board.result()
        with open('free_out.txt', 'a') as f:
            f.write(pgn + "\n")
        print("Completed game " + str(i))
        torch.save(model, "supervised_model.pt")

def get_sf_eval(position: chess.Board):
    stockfish.set_fen_position(position.fen())
    eval = stockfish.get_evaluation()
    if eval['type'] == 'mate':
        if eval['value'] > 0:
            return 1000
        else:
            return -1000
    else:
        return int(eval['value'])
    

model = torch.load("supervised_model.pt")

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
train_with_random_games(model, criterion, optimizer, 5, 6)
