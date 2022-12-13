import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_feats, h_feats, num_heads = 1)
        self.gat2 = GATConv(h_feats, num_classes, num_heads = 1)

    def forward(self, g, in_feat):
        h = self.gat1(g, in_feat)
        h = F.elu(h)
        h = self.gat2(g, h)
        return h
