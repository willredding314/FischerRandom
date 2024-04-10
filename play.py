import chess
import torch
from src.minimax import minimax_engine
from src.predictor import Predictor

torch.set_default_dtype(torch.float64)

model_path = "rand_games_linear_weights_inc.pt"
depth = 3

weights = torch.load(model_path)
play_model = Predictor()
play_model.fc1.weight = weights 


board = chess.Board(chess960=True)
pgn = ""

for i in range(1000):
    move, val = minimax_engine(board, depth - 1, 0, 1, board.turn, play_model)
    if move is None:
        break
    if i % 2 == 0:
        pgn += f'{i/2+1}. '
    print(board.san(move))
    pgn += f'{board.san(move)} '
    board.push(move)

pgn += board.result()
print(pgn)