import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class AttentionRecurrentConvModel(nn.Module):
    def __init__(self, input_size, n_classes, rnn_hidden_dim, cnn_kernel_size,cnn_output_channels,num_cnn_layers=1, stride=1, num_rnn_layers=1, bidirection=True, flag_cuda=False):

        '''
        Create a LSTM classifier with CNN local pattern detectors (1d).
        Take the weighted average of all hidden units for generating classes
        Arguments:
            input_size (tuple): A tuple of ints with (num of indices, time steps, feature_dimension)
            rnn_hidden_dim(int): Number of hidden nodes to use for recurrent layer
            cnn_kernel_size(int): Size of cnn kernels
            cnn_output_channels(int): Number of output channels of cnn layers
            num_cnn_layers (int): Number of cnn layers
            stride (int): Size of stride for CNN layers
            n_classes (int): Number of classes to score
            num_rnn_layers (int): Number of LSTM layers
            bidirection(bool): Bidirection LSTM or not
            flag_cuda(bool): Whether cuda support is enabled or not
        '''
        super(AttentionRecurrentConvModel, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_output_channels = cnn_output_channels
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_rnn_layers = num_rnn_layers
        self.num_cnn_layers = num_cnn_layers
        self.stride = stride
        self.bidirection = bidirection
        self.dropout_layer = nn.Dropout(p=0.2)
        if self.bidirection is True:
            self.num_direction = 2
        else:
            self.num_direction = 1
        self.cnn_module_list= nn.ModuleList(self.init_cnn_module_list())
        self.lstm_layer = nn.LSTM(input_size=int(input_size[-1]/(2**self.num_cnn_layers)), hidden_size=self.rnn_hidden_dim, num_layers=self.num_rnn_layers, bidirectional=self.bidirection, batch_first=True)
        self.fc_layer = nn.Linear(self.rnn_hidden_dim*self.num_direction, n_classes)
        self.attn = nn.Linear(self.rnn_hidden_dim * self.num_direction, 1, bias=False)
        self.flag_cuda = flag_cuda

    def init_hidden(self, batch_size):
        '''
        Init states
        :param batch_size:
        :return: initial hidden and cell states, both are tensors in (num_layers * num_directions, batch, hidden_size)
        '''
        h0, c0 = (Variable(torch.zeros(self.num_rnn_layers*self.num_direction, batch_size, self.rnn_hidden_dim)), Variable(torch.zeros(self.num_rnn_layers*self.num_direction, batch_size, self.rnn_hidden_dim)))
        if self.flag_cuda:
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def init_cnn_module_list(self):
        cnn_module_list = []
        prev_output_channels = 1
        for i in range(0,self.num_cnn_layers):
            num_channels = int(self.cnn_output_channels*(2**i))
            if num_channels > 0 and self.cnn_kernel_size %2 ==1:
                cnn = nn.Conv1d(prev_output_channels, num_channels, self.cnn_kernel_size, stride=self.stride,
                      padding=int((self.cnn_kernel_size - 1) / 2))
                cnn_module_list.append(cnn)
                cnn_module_list.append(nn.MaxPool1d(2,2))
                prev_output_channels = num_channels
            else:
                raise ValueError("Too many cnn layers!\n")
        return cnn_module_list


    def forward(self, x, lengths):

        x_flat = x.view(x.size()[0], 1, -1)
        for i in range(0, len(self.cnn_module_list), 2):
            x_flat = self.cnn_module_list[i](x_flat)
            x_flat = self.cnn_module_list[i + 1](F.leaky_relu(x_flat))

#TODO: debug
        x_flat = torch.mean(x_flat, dim=1).view(x_flat.size()[0], x.size()[1], -1)
        sequence = nn.utils.rnn.pack_padded_sequence(x_flat, lengths, batch_first=True)
        batch_size = len(x)
        h, c = self.init_hidden(batch_size)

        hx_seq, (hn, cn) = self.lstm_layer(sequence, (h, c))  # hx_seq ( N, T, D)
        output, length_list = nn.utils.rnn.pad_packed_sequence(hx_seq, batch_first=True)
        (N, T, D) = output.size()
        step_vector = self.attn(output.contiguous().view(-1, self.rnn_hidden_dim*self.num_direction))  # out_dim (N*T, 1)
        attn_weights = F.softmax(F.tanh(step_vector.view(-1, T)))  # out_dim (N, T)

        # create mask based on valid lengths
        mask = torch.ones(attn_weights.size())
        for i, l in enumerate(length_list):
            if l < T:
                mask[i, l:] = 0

        # apply mask and re-normalize attention weights
        mask = Variable(mask)
        if self.flag_cuda:
            mask = mask.cuda()
        masked = attn_weights * mask
        _sums = masked.sum(-1).unsqueeze(1).expand_as(attn_weights)
        attentions = masked.div(_sums)

        # Size of attn_weights/attentions should be (N, T, 1)
        temp_attn = attentions.view(-1, T, 1).repeat(1, 1, self.rnn_hidden_dim*self.num_direction)  # out_dim (N,T,1)=> (N,T,H)
        # Calculating weighted sum of hidden states
        attn_applied = torch.mul(temp_attn, output)  # out_dim (N,T,H)
        weighted = torch.sum(attn_applied, dim=1)
        score = self.fc_layer(self.dropout_layer(torch.squeeze(weighted, dim=1)))
        return score