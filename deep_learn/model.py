import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import shutil

class Model(nn.Module):
    """
    A class that defines the parameters for a
    sentence tagger of the form:

    INPUT --> EMBEDDING --> LSTM --> FC --> OUTPUT

    Parameters
        num_layers (int):   Number of LSTM Layers
        embedding_dim (int): Dimension of the embedding
        hidden_dim (int): Dimmension of the LSTM hidden layer
        batch_size (int): Batch size to process
        n_embeds (int): Size of the vocabulary
        output_dim (int):   Number of classes to predict
    """
    def __init__(self, num_layers, embedding_dim, hidden_dim,
                batch_size, n_embeds=64, output_dim=3):

        super(Model, self).__init__()
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size 
        self.n_embeds = n_embeds
        self.output_dim = output_dim

        self.word_embeddings = nn.Embedding(self.n_embeds, self.embedding_dim, 0)
        dp = 0 if self.num_layers == 1 else 0.5
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,
                            self.num_layers, dropout=dp)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.output_dim)

        self.best_dev = 0
        self.epoch = 0
        self.fname = 'n{}em{}hid{}bs{}best.pth.tar'.format(
                self.num_layers, self.embedding_dim, self.hidden_dim, self.batch_size)

    def init_hidden(self):
        """
        Returns the initial hidden state of the LSTM (zeros)
        """
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, sentence, lengths=None):
        """
        Arguments
            sentence (Tensor):  A tensor containing possibly padded sentences
            lengths (list): A list of lengths of each sentence

        Returns the models predicted tag probabilities
        """
        # Embed
        sentence = torch.t(sentence) # Transpose to fix the reshaping error
        embeds = self.word_embeddings(sentence)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths)

        # LSTM
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _= torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

        averaged = lstm_out.sum(0)
        averaged /= torch.tensor(lengths, dtype=torch.float).view(-1,1)
        # r = range(len(lengths))
        # averaged = lstm_out[lengths-1, r, :]

        tag_space = self.hidden2tag(averaged)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def save_checkpoint(self, o_state_dict, dev_acc, epoch):
        """
        Saves the model as a checkpoint. The filename is
        self.fname

        Arguments
            o_state_dict (dict):    The optimizer's state_dict)
            dev_acc (float):    Accuracy on the development set
            epoch (int):    the epoch number.

        Returns None
        """
        state = {
            'epoch': epoch + 1,
            'state_dict': self.state_dict(),
            'best_dev': max(dev_acc, self.best_dev),
            'optimizer' : o_state_dict,
        }

        torch.save(state, 'checkpoint.pth.tar')
        if dev_acc > self.best_dev:
            shutil.copyfile('checkpoint.pth.tar', self.fname)
            self.best_dev = dev_acc

    def predict(self, sentences, word2ind):
        """
        Predict the tags of each sentence in sentence

        Arguments
            sentences (list of str): Sentences to be predicted
            word2ind (dict): Mapping from words to indices in the vocab

        Returns
            (list) Predicted tags as strings
        """
        self.eval()
        preprocessed = [preprocess(s.text) for s in sentences]
        preprocessed = [' '.join([str(word2ind[w] if w in word2ind else '1') for w in s.split()]) for s in preprocessed]
        self.batch_size = len(preprocessed)

        self.hidden = self.init_hidden()
        model_in, lens, _, sort = prepare_sentence(preprocessed, None, self.batch_size, True)
        inverse_sort = sort.argsort()

        activation = self.forward(model_in, lens).data.numpy()
        preds = activation.argmax(1)[inverse_sort] # Need to inverse sort b/c I sorted during prepare

        labels = np.array([2,3,4])
        return labels[preds]
