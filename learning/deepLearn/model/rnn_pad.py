import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, n_classes, hidden_dim, num_layers=1, bidirection=1, flag_cuda=False):
        '''
        Create a LSTM classifier initialize their weights.
        Take the last hidden unit for generating classes
        Arguments:
            im_size (tuple): A tuple of ints with (num of indices, time steps, feature_dimension)
            hidden_dim (int): Number of hidden nodes to use
            n_classes (int): Number of classes to score
            num_layers (int): Number of LSTM layers
            bidirection(int): Bidirection LSTM or not
            flag_cuda(bool): Whether cuda support is enabled or not, 1 is true
        '''
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirection = bidirection
        if self.bidirection == 1:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.lstm_layer = nn.LSTM(input_size=input_size[-1], hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirection, batch_first=True)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc_layer = nn.Linear(hidden_dim*self.num_direction, n_classes)
        self.flag_cuda = flag_cuda

    def init_hidden(self, batch_size):
        '''
        Init states
        :param batch_size:
        :return: initial hidden and cell states, both are tensors in (num_layers * num_directions, batch, hidden_size)
        '''
        h0, c0 = (Variable(torch.zeros(self.num_direction*self.num_layers, batch_size, self.hidden_dim)), Variable(torch.zeros(self.num_direction*self.num_layers, batch_size, self.hidden_dim)))
        if self.cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, x, lengths):
        batch_size = len(x)
        h, c = self.init_hidden(batch_size)
        sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        output, (hn, cn) = self.lstm_layer(sequence, (h, c))
        score = self.fc_layer(self.dropout_layer(hn[-1,:,:]))
        return score

