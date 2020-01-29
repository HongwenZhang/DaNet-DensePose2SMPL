# The based unit of graph convolutional networks.
# code brought in part from https://github.com/RexYing/diffpool/blob/master/encoders.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.graph import normalize_undigraph


# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
            #print(y[0][0])
        return y


class GCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, out_dim, num_layers, num_nodes, bn=True, normalize=False, learn_edge=False, edge_init=1, edge_act_fun=F.relu, dropout=0):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.learn_edge = learn_edge

        self.edge_act = edge_act_fun

        self.gc = nn.ModuleList()
        self.act = nn.ModuleList()

        if self.num_layers == 1:
            self.gc.append(GraphConv(input_dim, out_dim, normalize_embedding=normalize))
        else:
            self.gc.append(GraphConv(input_dim, hidden_dim, normalize_embedding=normalize))
            for i in range(self.num_layers-2):
                self.gc.append(GraphConv(hidden_dim, hidden_dim, normalize_embedding=normalize))
            self.gc.append(GraphConv(hidden_dim, out_dim, normalize_embedding=normalize))

        for i in range(self.num_layers):
            if bn:
                self.act.append(nn.Sequential(nn.BatchNorm1d(num_nodes), nn.ReLU(inplace=True)))
            else:
                self.act.append(nn.ReLU(inplace=True))

        if self.learn_edge:
            self.register_buffer('I_n', torch.eye(num_nodes).float())
            self.mask = nn.ParameterList()
            for i in range(self.num_layers):
                self.mask.append(nn.Parameter(edge_init * torch.ones((1, num_nodes, num_nodes))))

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = nn.init.constant(m.bias.data, 0.0)

    def forward(self, x, A):

        hidden = x
        for i in range(self.num_layers):
            if self.learn_edge:
                hidden = self.act[i](self.gc[i](hidden, normalize_undigraph(self.I_n + A * self.edge_act(self.mask[i]))[0]))
            else:
                hidden = self.act[i](self.gc[i](hidden, A))

        return hidden

