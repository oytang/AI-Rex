import math
import torch
import torch.nn as nn
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import ModuleList
import torch.nn.functional as F
from layer.graph_transformer_edge_layer import GraphTransformerLayer


class GNN(nn.Module):
    def __init__(self, gnn, n_layer, node_dim, edge_dim, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.feature_len = node_dim
        self.dim = dim
        self.gnn_layers = ModuleList([])
        if gnn in ['gcn', 'gat', 'sage', 'tag', 'gt']:
            for i in range(n_layer):
                if gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=self.feature_len if i == 0 else dim,
                                                     out_feats=dim,
                                                     activation=None if i == n_layer - 1 else torch.relu))
                elif gnn == 'gat':
                    num_heads = 16  # make sure that dim is dividable by num_heads
                    self.gnn_layers.append(GATConv(in_feats=self.feature_len if i == 0 else dim,
                                                   out_feats=dim // num_heads,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=self.feature_len if i == 0 else dim,
                                                    out_feats=dim,
                                                    activation=None if i == n_layer - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=self.feature_len if i == 0 else dim,
                                                   out_feats=dim,
                                                   activation=None if i == n_layer - 1 else torch.relu,
                                                   k=hops))
                elif gnn == 'gt':
                    self.node_emb = nn.Linear(node_dim, dim)
                    self.edge_emb = nn.Linear(edge_dim, dim)
                    self.graph_transformer = GraphTransformerLayer(dim, dim, num_heads=4, dropout=0.2)
                    self.gnn_layers.append(self.graph_transformer)


        elif gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=self.feature_len, out_feats=dim, k=n_layer))
        else:
            raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()
        self.factor = None

    def forward(self, graph):
        h = graph.ndata['feature']
        e = graph.edata['feature']
        if self.gnn == 'gt':
            h = self.node_emb(h)
            e = self.edge_emb(e)
            for layer in self.gnn_layers:
                h, e = layer(graph, h, e)
        else:
            for layer in self.gnn_layers:
                h = layer(graph, h)
                if self.gnn == 'gat':
                    h = torch.reshape(h, [h.size()[0], -1])
        if self.factor is None:
            self.factor = math.sqrt(self.dim) / float(torch.mean(torch.linalg.norm(h, dim=1)))
        h *= self.factor
        graph_embedding = self.pooling_layer(graph, h)
        return graph_embedding


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2): #L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [ nn.Linear( input_dim//2**l , input_dim//2**(l+1) , bias=True ) for l in range(L) ]
        list_FC_layers.append(nn.Linear( input_dim//2**L , output_dim , bias=True ))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L
        
    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return torch.sigmoid(y)


class MyModel(nn.Module):
    def __init__(self, gnn, n_layer, node_dim, edge_dim, dim):
        super(MyModel, self).__init__()
        self.GNN = GNN(gnn, n_layer, node_dim, edge_dim, dim)
        self.readout = MLPReadout(dim*2, 1)

    def forward(self, reactant_graph, product_graph):
        reactant_emb = self.GNN(reactant_graph)
        product_emb = self.GNN(product_graph)
        all_emb = torch.cat([reactant_emb, product_emb], dim=1)
        out = self.readout(all_emb)

        return out
