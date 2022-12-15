import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import Sequential
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, args, in_feats, num_classes=None):
        super(GCN, self).__init__()
        dropout = args.dropout
        
        hidden_units = args.hidden_units
        if not isinstance(hidden_units, list):
            hidden_units = [hidden_units]
            
        # if not isinstance(dropout, list):
        #     dropout = [dropout] * len(hidden_units)
        
        self.hidden_units = [in_feats] + hidden_units
        self.gconv_layers = nn.ModuleList()
        
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        
        for i in range(len(self.hidden_units)-1):
            self.gconv_layers.append(GraphConv(self.hidden_units[i], self.hidden_units[i+1], activation=F.relu))
            
        # Last layer
        if num_classes:
            self.gconv_layers.append(GraphConv(self.hidden_units[-1], num_classes))
            

    def forward(self, g, in_feat):
        h = in_feat
        for i, gconv in enumerate(self.gconv_layers):
            h = gconv(g, h)
            if self.dropout and i != len(self.gconv_layers)-1:
                h = self.dropout(h)

        return h