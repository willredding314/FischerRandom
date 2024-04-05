import copy
import chess
import torch
from src.minimax import minimax_engine


def run_game(board, model):
    positions_for_white = []
    positions_for_black = []
    white_to_move = True
    
    while True:
        move, val = minimax_engine(board, 1, 0, 1, board.turn, model)
        if move is None:
            break
        board.push(move)
        if white_to_move:
            positions_for_white.append(board.copy())
        else:
            positions_for_black.append(board.copy())
        white_to_move = not white_to_move

    return positions_for_white, positions_for_black, not white_to_move

def train_with_random_games(model, criterion, optimizer, num_games, depth):
    for i in range(num_games):
        board = chess.Board(chess960=True)
        white_positions, black_positions, white_wins = run_game(board, copy.deepcopy(model))

        white_target = 0
        if white_wins:
            white_target = 1
    
        for position in white_positions:
            optimizer.zero_grad()
            mv, value = minimax_engine(position, depth - 1, 0, 1, True, model)
            if mv is None:
                break
            loss = criterion(value, torch.as_tensor([white_target], dtype=torch.float64))
            loss.backward()
            optimizer.step()

        for position in black_positions:
            optimizer.zero_grad()
            mv, value = minimax_engine(position, depth - 1, 0, 1, False, model)
            if mv is None:
                break
            loss = criterion(value, torch.as_tensor([1 - white_target], dtype=torch.float64))
            loss.backward()
            optimizer.step()

        print("Completed game " + str(i))
        torch.save(model.fc1.weight, "rand_games_linear_weights2.pt")