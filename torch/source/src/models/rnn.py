import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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


# class BasicRNN(nn.Module):
#     def __init__(self, max_word_len, hidden_size, output_size, vocab_len):
#         super().__init__()
#         embedding_dim = 10
#         self.max_word_len = max_word_len
#         self.embed = nn.Embedding(vocab_len, embedding_dim)
#         self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
#         self.i2o = nn.Linear(max_word_len*hidden_size, output_size)

#     def forward(self, x):
#         x, seq_lengths = x
#         batch_size, _ = x.shape
#         seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#         out = x[perm_idx]
#         out = self.embed(out)
#         out = pack_padded_sequence(out, seq_lengths, batch_first=True, enforce_sorted=True)
#         out, _ = self.rnn(out)
#         out, input_sizes = pad_packed_sequence(out, batch_first=True,
# total_length=self.max_word_len)
#         out = out.reshape((batch_size, -1))
#         out = self.i2o(out)
#         return out
