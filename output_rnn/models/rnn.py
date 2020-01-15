import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
        # seq_tensor = seq_tensor[perm_idx]
        out, _ = self.rnn(x)
        out = out.squeeze()
        out = self.i2o(out)
        return out
