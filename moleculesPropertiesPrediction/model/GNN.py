import torch
from torch_geometric.nn import GCNConv, global_mean_pool, TransformerConv, GATv2Conv
import torch.nn.functional as Fun


class GCN(torch.nn.Module):
    def __init__(self, dim_h,qm9_num_features, targets_num=1):
        super().__init__()
        self.conv1 = GCNConv(qm9_num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = torch.nn.Linear(dim_h, targets_num)

    def forward(self, data):
        x, e = data.x, data.edge_index

        x = self.conv1(x, e)
        x = x.relu()
        x = self.conv2(x, e)
        x = x.relu()
        x = self.conv3(x, e)

        x = global_mean_pool(x, data.batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x

class TransformerCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=4)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4)
        self.conv3 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4)
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x

class Gatv2CN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=4, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4, concat=True)
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x