import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.nn import init
from graph_conv import calculate_laplacian_with_self_loop2
import time

class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, windd, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.adj = torch.FloatTensor(adj).cuda()
        self.windmap = torch.tensor(windd).cuda()
        self.liner1 = nn.Linear(self._num_gru_units + 1, self._output_dim)
        self.liner2= nn.Linear(self.adj.shape[0], self.adj.shape[0])
        self.liner3 = nn.Linear(self.adj.shape[0], self.adj.shape[0])

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, _, num_nodes = inputs.shape

        ####direction
        wind_now = inputs[:, 2, :] * 360
        wind_now = wind_now.unsqueeze(2)
        wind_now_map = wind_now.repeat(1, 1, num_nodes)
        windlabel = self.windmap.unsqueeze(0).repeat(batch_size, 1, 1)
        direction = (windlabel - wind_now_map).abs()
        direction[direction > 360] = 180
        direction1 = torch.cos(direction * 1 / 180 * np.pi)
        direction1[direction1 < 0] = 0
        ###########speed######
        # speed_now = inputs[:, 1, :]
        # speed_now = speed_now.unsqueeze(2)
        # speed_now= speed_now.repeat(1, 1, 67)

        a = self.liner2(direction1)+self.liner3(self.adj)
        a[a<0]=0
        a1 = calculate_laplacian_with_self_loop2(a)
        inputs = inputs[:, 0, :].reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        a_times_concat = a1 @ concatenation
        outputs = self.liner1(a_times_concat)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1   = nn.Conv1d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv1d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int, windd):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.windd = windd
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim * 2, self.windd, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolution(
            self.adj, self._hidden_dim, self._hidden_dim, self.windd
        )
        self.ca1 =ChannelAttention(self._hidden_dim*2)
        self.sa1 = SpatialAttention()
    def forward(self, inputs, hidden_state):
        concatenation = self.graph_conv1(inputs, hidden_state)
        concatenation = concatenation.reshape(-1,(2 * self._hidden_dim),self._input_dim)
        concatenation = self.ca1(concatenation)*concatenation
        concatenation = self.sa1(concatenation) * concatenation
        concatenation = concatenation.reshape(-1, (2 * self._hidden_dim)* self._input_dim)
        concatenation = torch.sigmoid(concatenation)
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        com = self.graph_conv2(inputs, r * hidden_state)
        c = torch.tanh(com)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state



class ass_TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int, pre_len, wind, **kwargs):
        super(ass_TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.wind = wind
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim, self.wind)
        self.tgcn_cell1 = TGCNCell(self.adj, self._input_dim, self._hidden_dim, self.wind)
        self.tgcn_cell2 = TGCNCell(self.adj, self._input_dim, self._hidden_dim, self.wind)
        self.tgcn_cell3 = TGCNCell(self.adj, self._input_dim, self._hidden_dim, self.wind)
        self.linear1 = torch.nn.Linear(hidden_dim, 1)
        self.linear2 = torch.nn.Linear(hidden_dim, 1)
        self.linear3 = torch.nn.Linear(hidden_dim, 1)
        self.linearo = torch.nn.Linear(hidden_dim, pre_len)
    def forward(self, inputs):
        batch_size, seq_len, _, num_nodes = inputs.shape
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs
        )
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            output = self.linear1(output)
            output = output.permute([0, 2, 1])
            input2 = inputs[:, i, :]
            input2[:,0:1,:] = inputs[:,i, 0:1, :]+output
            output, hidden_state = self.tgcn_cell1(input2, hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            output = self.linear2(output)
            output = output.permute([0, 2, 1])
            input3 = inputs[:, i, :]
            input3[:,0:1,:] = inputs[:,i, 0:1, :]+output
            output, hidden_state = self.tgcn_cell2(input3, hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            output = self.linear3(output)
            output = output.permute([0, 2, 1])
            input4 = inputs[:, i, :]
            input4[:, 0:1, :] = inputs[:, i, 0:1, :] + output
            output, hidden_state = self.tgcn_cell3(input4, hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        outputs = self.linearo(output)
        outputs = outputs.permute([0, 2, 1])
        return outputs

