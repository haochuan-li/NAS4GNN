import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(
        self,
        args,
        in_feats,
        num_classes=None,
    ):
        super(APPNP, self).__init__()

        hidden_units = args.hidden_units
        if not isinstance(hidden_units, list):
            hidden_units = [hidden_units]

        self.hidden_units = [in_feats] + hidden_units 
        self.layers = nn.ModuleList()

        for i in range(len(self.hidden_units)-1):
            self.layers.append(nn.Linear(self.hidden_units[i], self.hidden_units[i+1]))
            
        # Last layer
        if num_classes:
            self.layers.append(nn.Linear(self.hidden_units[-1], num_classes))
         
        self.activation = F.relu 
        if args.dropout:
            self.feat_drop = nn.Dropout(args.dropout)
        else:
            self.feat_drop = lambda x: x
        self.propagate = APPNPConv(args.k, args.alpha, args.edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, g, in_feat):
        # prediction step
        h = in_feat
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))

        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))

        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(g, h)
        return h
