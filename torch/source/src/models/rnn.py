import torch.nn as nn


class BasicRNN(nn.Module):
    def __init__(self, input_shape, max_word_len, hidden_size, output_size):
        super().__init__()
        self.input_shape = input_shape
        self.max_word_len = max_word_len
        self.rnn = nn.RNN(max_word_len, hidden_size, batch_first=True)
        self.i2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.squeeze()
        x = self.i2o(x)
        return x
