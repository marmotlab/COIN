from torch import nn
import torch
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    Graph Attention Layer
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features  # Number of input features for each node
        self.out_features = out_features  # Number of output features for each node
        self.dropout = dropout  # Dropout parameter
        self.alpha = alpha  # LeakyReLU activation parameter
        self.concat = concat  # If true, apply ELU activation afterward

        # Define trainable parameters, i.e., W and a in the paper
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Initialization
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Initialization

        # Define LeakyReLU activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp, adj):
        """
        inp: input_fea [B, N, in_features] where in_features is the number of elements in the input feature vector for each node
        adj: adjacency matrix of the graph [N, N], nonzero means there is a connection
        input: (B, N, C_in)
        output: (B, N, C_out)
        """
        # [B, N, out_features]
        h = torch.matmul(inp, self.W)
        N = h.size()[1]  # Number of nodes in the graph

        # [B, N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, 1, N).view(-1, N * N, self.out_features), h.repeat(1, N, 1)], dim=-1).view(-1,
                                                                                                                    N,
                                                                                                                    N,
                                                                                                                    2 * self.out_features)

        # [B, N, N, 1] => [B, N, N] Un-normalized attention coefficients of the graph attention
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # Set unconnected edges to negative infinity
        zero_vec = -1e12 * torch.ones_like(e)

        adj = adj.unsqueeze(-1)
        adj.expand(adj.size(0), adj.size(1), adj.size(1))

        # [B, N, N]
        attention = torch.where(adj < 1, e, zero_vec)
        # Indicates that if an element in the adjacency matrix is 0, then there is a connection between the nodes,
        # and the attention coefficient is retained, otherwise, it needs to be masked and set to a very small value,
        # because this minimum value will be ignored during softmax.
        # Shape remains unchanged [B, N, N], obtain normalized attention weights!
        attention = F.softmax(attention,
                              dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # [B, N, N].[B, N, out_features] => [B, N, out_features]
        h_prime = torch.matmul(attention, h)
        # Get the representation updated by the surrounding nodes through attention weights
        if self.concat:
            return F.relu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, n_heads, dropout=0, alpha=0.2):
        """
        Dense version of GAT
        n_heads indicates how many GAL layers there are, which are concatenated together in the end,
        similar to self-attention to extract features from different subspaces.
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        # Define multi-head graph attention layers
        self.attentions = [GraphAttentionLayer(n_feat, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)  # Add to pytorch's Module
        # The output layer is also implemented through a graph attention layer, which can be used for classification, prediction, etc.
        self.out_att = GraphAttentionLayer(n_hid * n_heads, n_class, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions],
                      dim=2)  # Concatenate the representations obtained from each head
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))  # Output and activate
        return F.log_softmax(x, dim=2)  # log_softmax for faster speed and numerical stability


if __name__ == '__main__':
    s = torch.randn(5, 10, 64)
    a = torch.zeros(5, 10, 10)

    net = GAT(n_feat=64, n_hid=64, n_class=64, alpha=0.1, n_heads=4)
    out = net(s, a)

    print(out.shape)
