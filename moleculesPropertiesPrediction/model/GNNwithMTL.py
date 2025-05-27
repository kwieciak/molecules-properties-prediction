import torch
from torch_geometric.nn import GCNConv, TransformerConv, GATv2Conv, GINConv,  global_mean_pool, global_add_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, ModuleDict, ModuleList
import torch.nn.functional as Fun


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.task_heads = ModuleList({
            Linear(hidden_channels, out_channels) for _ in num_tasks
        })

    def forward(self, data):
        x, edge_index, r_target  = data.x, data.edge_index, data.r_target

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, data.batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        outs = [head(x) for head in self.task_heads]
        return torch.cat(outs, dim=1)

class TransformerCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, edge_dim=4)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4)
        self.conv3 = TransformerConv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4)

        self.task_heads = ModuleList({
            Linear(hidden_channels, out_channels) for _ in num_tasks
        })

    def forward(self, data):
        x, edge_index, edge_attr, batch, r_target = data.x, data.edge_index, data.edge_attr, data.batch, data.r_target

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        outs = [head(x) for head in self.task_heads]
        return torch.cat(outs, dim=1)

class Gatv2CN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks, out_channels=1, heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=4, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=4, concat=True)

        self.task_heads = ModuleList({
            Linear(hidden_channels, out_channels) for _ in num_tasks
        })

    def forward(self, data):
        x, edge_index, edge_attr, batch, r_target = data.x, data.edge_index, data.edge_attr, data.batch, data.r_target

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)

        x = global_mean_pool(x, batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        outs = [head(x) for head in self.task_heads if r_target ]
        return torch.cat(outs, dim=1)


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_tasks, out_channels=1):
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

        self.task_heads = ModuleDict({
            str(i): Linear(hidden_channels, out_channels)
            for i in num_tasks
        })

    def forward(self, data):
        x, edge_index, batch, r_targets = data.x, data.edge_index, data.batch, data.r_target

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        outs = []
        for i, r_target in enumerate(r_targets):
            head = self.task_heads[str(int(r_target))]
            outs.append(head(x[i].unsqueeze(0)))

        return torch.cat(outs, dim=0).squeeze(-1)

        #outs = [head(x) for head in self.task_heads]
        #return torch.cat(outs, dim=1) <- stare podejscie


