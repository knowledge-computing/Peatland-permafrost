import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, x):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * x)
    
def get_nonlinear_layer(nonlinearity):
    if nonlinearity == 'identity':
        return nn.Identity()
    elif nonlinearity == 'sine':
        return Sine()
    elif nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'selu':
        return nn.SELU(inplace=True)
    elif nonlinearity == 'softplus':
        return nn.Softplus()
    elif nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif nonlinearity == 'gelu':
        return nn.GELU()
    elif nonlinearity == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dims,
        nonlinearity='relu',
    ):
        super().__init__()        
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_nonlinear_layer(nonlinearity))            
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)
    
    
class ResLayer(nn.Module):
    """
        Code Reference: https://github.com/elijahcole/sinr/blob/main/models.py
    """
    def __init__(
        self, 
        in_dim,
        dropout_rate=0,
        skip=False,
        norm=False,
    ):
        super(ResLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, in_dim)
        self.lin2 = nn.Linear(in_dim, in_dim)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        y = self.lin1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.lin2(y)
        y = self.nonlin2(y)
        return x + y

    
class FCLayer(nn.Module):
    """
        Code Reference: https://github.com/gengchenmai/csp/blob/main/main/module.py
    """
    def __init__(
        self, 
        in_dim,
        out_dim,
        nonlinearity,
        dropout_rate=0.,
        layer_norm=False,
        skip=False
    ):
        super(FCLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skip_connection = skip
        self.lin = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform(self.lin.weight)
        self.nonlin = get_nonlinear_layer(nonlinearity)

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        if layer_norm:
            self.layer_norm = nn.LayerNorm(self.out_dim)
        else:
            self.layer_norm = nn.Identity()
                
    def forward(self, x):
        y = self.lin(x)
        y = self.nonlin(y)
        y = self.dropout(y)
        if self.skip_connection:
            y = y + x
        y = self.layer_norm(y)
        return y
    