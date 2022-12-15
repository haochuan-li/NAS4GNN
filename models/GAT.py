import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, args, in_feats, num_classes=None):
        super(GAT, self).__init__()
        heads = args.heads
        hidden_units = args.hidden_units
        
        if not isinstance(hidden_units, list):
            hidden_units = [hidden_units]
        
        hidden_units = [in_feats] + hidden_units
        
        if not isinstance(heads, list):
            heads = [heads]
        
        heads.append(1)
        
        self.gat_layers = nn.ModuleList()
        
        assert len(heads) == len(hidden_units)
        
        for i in range(len(hidden_units)-1):
            if i == 0:
                self.gat_layers.append(GATConv(hidden_units[i], hidden_units[i+1], heads[i], feat_drop=0.6, attn_drop=0.6, activation=F.elu))
            else:
                self.gat_layers.append(GATConv(hidden_units[i]*heads[i-1], hidden_units[i+1], heads[i], feat_drop=0.6, attn_drop=0.6, activation=F.elu))
        if num_classes:
            self.gat_layers.append(GATConv(hidden_units[-1]*heads[-2], num_classes, heads[-1], feat_drop=0.6, attn_drop=0.6))

    def forward(self, g, in_feat):
        h = in_feat
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == len(self.gat_layers)-1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)
                
        return h