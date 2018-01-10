import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class AttentionRecurrentModel(nn.Module):
    def __init__(self, input_size, n_classes, hidden_dim, attn_dim=60, num_layers=1, bidirection=True, flag_cuda=False):
        '''
            Create a LSTM classifier initialize their weights.
            Take the weighted average of all hidden units for generating classes
            Arguments:
                input_size (tuple): A tuple of ints with (num of indices, time steps, feature_dimension) => (N, T, D)
                hidden_dim (int): Number of hidden activations to use
                n_classes (int): Number of classes to score
                attn_dim (int): Number of hidden nodes in attention layers
                num_layers (int): Number of LSTM layers
                bidirection(bool): Bidirection LSTM or not, for attention model the default value is True
                flag_cuda(bool): Whether cuda support is enabled or not
        '''
        super(AttentionRecurrentModel, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.num_layers = num_layers
        self.bidirection = bidirection
        if self.bidirection is True:
            self.num_direction = 2
        else:
            self.num_direction = 1
        # set batch_first to make input and output in tensor of (N,T,D) but not h0, c0
        self.lstm_layer = nn.LSTM(input_size=input_size[-1], hidden_size=hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirection)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.fc_layer = nn.Linear(hidden_dim*self.num_direction , n_classes)
        self.attn = nn.Linear(self.hidden_dim*self.num_direction, 1, bias=False)
        self.flag_cuda = flag_cuda


    def init_hidden(self, batch_size):
        '''
        Init states
        :param batch_size (int)
        :return: initial hidden and cell states, both are tensors in (num_layers * num_directions, batch, hidden_size)
        '''
        h0, c0 = (Variable(torch.zeros(self.num_layers*self.num_direction, batch_size, self.hidden_dim)), Variable(torch.zeros(self.num_layers*self.num_direction, batch_size, self.hidden_dim)))
        if self.flag_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, x, lengths):
        batch_size = len(x)
        h, c = self.init_hidden(batch_size)
        sequence = nn.utils.rnn.pack_padded_sequence(x.float(), lengths, batch_first=True)
        hx_seq, (hn, cn) = self.lstm_layer(sequence, (h, c)) #hx_seq ( N, T, D)
        output, length_list = nn.utils.rnn.pad_packed_sequence(hx_seq, batch_first=True)
        (N, T, D) = output.size()
        step_vector = self.attn(output.contiguous().view(-1, self.hidden_dim*self.num_direction))  # out_dim (N*T, 1)
        attn_weights = F.softmax(F.tanh(step_vector.view(-1, T))) # out_dim (N, T)

        # create mask based on valid lengths
        mask = torch.ones(attn_weights.size())
        for i, l in enumerate(length_list):
            if l < T:
                mask[i, l:] = 0

        # apply mask and re-normalize attention weights
        mask = Variable(mask)
        if self.flag_cuda:
            mask = mask.cuda()
        masked = attn_weights*mask
        _sums = masked.sum(-1).unsqueeze(1).expand_as(attn_weights)
        attentions = masked.div(_sums)

        # Size of attn_weights/attentions should be (N, T, 1)
        temp_attn =attentions.view(-1, T, 1).repeat(1,1,self.hidden_dim*self.num_direction) # out_dim (N,T,1)=> (N,T,H)
        # Calculating weighted sum of hidden states
        attn_applied = torch.mul(temp_attn, output)  # out_dim (N,T,H)
        weighted = torch.sum(attn_applied, dim=1)
        score = self.fc_layer(self.dropout_layer(torch.squeeze(weighted, dim=1)))
        return score
