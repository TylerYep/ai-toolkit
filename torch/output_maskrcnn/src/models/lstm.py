import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class BasicLSTM(nn.Module):
    def __init__(self, input_shape, vocab_size, output_size, embedding_size=128, hidden_size=32):
        super().__init__()
        self.input_shape = input_shape
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=1)

        self.hidden2out = nn.Linear(hidden_size, output_size)
        self.dropout_layer = nn.Dropout(p=0.2)

    def init_hidden(self, batch_size):
        return torch.randn(1, batch_size, self.hidden_size), \
            torch.randn(1, batch_size, self.hidden_size)

    def forward(self, batch, lengths):
        hidden = self.init_hidden(batch.size(-1))
        embeds = self.embedding(batch)
        packed_input = pack_padded_sequence(embeds, lengths)
        _, (ht, _) = self.lstm(packed_input, hidden)

        # ht is the last hidden state of the sequences
        # ht = (1 x batch_size x hidden_size)
        # ht[-1] = (batch_size x hidden_size)
        output = self.dropout_layer(ht[-1])
        output = self.hidden2out(output)
        output = F.log_softmax(output, dim=1)
        return output
