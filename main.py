import json
import chess
import torch
from src.concepts.eval import value_tensor
from src.predictor import Predictor
from src.train import train_with_random_games

torch.set_default_dtype(torch.float64)
model = Predictor()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters())

train_with_random_games(model, criterion, optimizer, 1, 6)