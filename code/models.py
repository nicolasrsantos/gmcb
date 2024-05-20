import torch.nn as nn
from torch.nn import functional as F
import torch_geometric.nn as tgnn

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.conv1 = tgnn.SAGEConv(in_channels, hidden_channels)
        self.conv2 = tgnn.SAGEConv(hidden_channels, hidden_channels)
        self.classifier = tgnn.Linear(hidden_channels, out_channels)

    def forward(self, x_doc, x_token, edge_index, edge_weight):
        h = self.conv1(x_doc, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        return self.classifier(h), h
