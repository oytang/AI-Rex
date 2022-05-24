# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:34:56 2022

@author: wanxiang.shen@u.nus.edu


"""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Sequential, Linear, ModuleList, ReLU

from torch_geometric.nn import MessagePassing, JumpingKnowledge
from torch_geometric.nn import NNConv, GATv2Conv, PNAConv, SAGEConv, GINEConv, MLP 
from torch_geometric.nn import global_mean_pool, global_max_pool, Set2Set, GlobalAttention

import os


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, data_transformer, 
                 save_dir = './outputs', 
                 save_name = 'best_model.pt', 
                 best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        self.save_name = save_name
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.inMemorySave = {'data_transformer': data_transformer}
        
        
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            # print(f"\nBest validation loss: {self.best_valid_loss}")
            # print(f"\nSaving best model for epoch: {epoch+1}\n")
            self.inMemorySave.update({'epoch': epoch+1, 'model_args':model.model_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()})
            

    def save(self):
        print(f"Saving final model...")
        print(f"\nBest validation loss: {self.best_valid_loss}")
        print(f"\nSaving best model for epoch: {self.inMemorySave['epoch']}\n")
            
        torch.save(self.inMemorySave, os.path.join(self.save_dir,
                                                   self.save_name))


class BVNet(torch.nn.Module):

    r"""GNN-based Atomic Buried  Volume Prediction Model
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each out sample.
        edge_dim (int): Edge feature dimensionality. 
        hidden_channels (int, optional): Size of each hidden sample. (default: :int:64)
        num_layers (int, optional): Number of message passing layers. (default: :int:2)
        dropout_p (float, optional): Dropout probability. (default: :obj:`0.1`) of ACNet, different from dropout in GATConv layer
        batch_norms (torch.nn.Module, optional, say torch.nn.BatchNorm1d): The normalization operator to use. (default: :obj:`None`)
        global_pool: (torch_geometric.nn.Module, optional): the global-pooling-layer. (default: :obj: torch_geometric.nn.global_mean_pool)
        **kwargs (optional): Additional arguments of the underlying:class:`torch_geometric.nn.conv.MessagePassing` layers.
    """
    
    
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 edge_dim,
                 hidden_channels = 64, 
                 num_layers = 2,
                 dropout_p = 0.1,
                 batch_norms = None,
                 global_pool = global_mean_pool,
                 **kwargs,
                ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.jk_mode = 'cat'
        self.batch_norms = batch_norms
        self.global_pool = global_mean_pool
        
        model_args = {'in_channels':self.in_channels, 
                'hidden_channels':self.hidden_channels, 
                'out_channels':self.out_channels,
                'edge_dim':self.edge_dim, 
                'num_layers': self.num_layers, 
                'dropout_p':self.dropout_p, 
                'batch_norms':self.batch_norms,
                'global_pool':self.global_pool
               }
        for k, v in kwargs.items():
            model_args[k] = v
        self.model_args = model_args
        
        
        ## layer stack 
        self.convs = ModuleList()
        self.convs.append(self.init_conv(in_channels, hidden_channels, edge_dim, **kwargs))
        for _ in range(num_layers - 1):
            self.convs.append(self.init_conv(hidden_channels, hidden_channels, edge_dim, **kwargs))
        
        # norm stack
        if batch_norms is not None:
            self.batch_norms = ModuleList()
            for _ in range(num_layers):
                self.batch_norms.append(copy.deepcopy(batch_norms))

        self.jk = JumpingKnowledge(self.jk_mode, hidden_channels, num_layers)
        self.lin = Linear(num_layers * hidden_channels, self.out_channels)
        

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.batch_norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()
            

    def init_conv(self, in_channels, out_channels, edge_dim, **kwargs): 
        #False concat the head, to average the information
        concat = kwargs.pop('concat', False)
        return GATv2Conv(in_channels, out_channels, edge_dim = edge_dim, concat=concat, **kwargs)
        
        
        
    def forward(self, x, edge_index, edge_attr, batch,  *args, **kwargs):

        x = F.dropout(x, p=self.dropout_p, training = self.training)
        
        # conv-act-norm-drop layer
        xs = []  
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr, *args, **kwargs)        
            x = F.relu(x, inplace=True)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
            xs.append(x) #for jk
            
        # the jk layer        
        x = self.jk(xs) #64*2
        
        # global pooling layer, please replace it with fuctinal group pooling @cuichao
        #embed = self.global_pool(x, batch)
        
        # output
        y = self.lin(x)
        return y
    
    
    
