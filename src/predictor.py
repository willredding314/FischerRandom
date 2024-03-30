import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):

    def __init__(self):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(100, 1) # GET MORE ACCURATE VALUE

    def forward(self, x):
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
    
    