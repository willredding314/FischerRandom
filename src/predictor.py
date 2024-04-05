import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(11, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sig(x)
        return x
    
    