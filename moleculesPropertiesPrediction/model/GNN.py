import torch
import torch.nn.functional as Fun
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GCNConv, TransformerConv, GATv2Conv, GINConv, global_mean_pool, global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

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
        x = Fun.dropout(x, p=0.35, training=self.training)

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
        x = Fun.dropout(x, p=0.35, training=self.training)

        x = self.lin(x)

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(in_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU()
            )
        )
        self.conv2 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU()
            )
        )
        self.conv3 = GINConv(
            Sequential(
                Linear(hidden_channels, hidden_channels),
                BatchNorm1d(hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels),
                ReLU()
            )
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)
        x = Fun.dropout(x, p=0.35, training=self.training)

        x = self.lin(x)

        return x
