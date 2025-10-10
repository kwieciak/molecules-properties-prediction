import torch
import torch.nn.functional as Fun
from torch import nn
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, ModuleDict
from torch_geometric.nn import GCNConv, TransformerConv, GATv2Conv, global_mean_pool, GINConv


class BaseMTLGNN(nn.Module):
    def __init__(self, tasks, hidden_dim, out_dim=1, head_hidden_dim=None, dropout_p=None,
                 pooling_fn=global_mean_pool):
        super().__init__()
        self.dropout_p = dropout_p
        self.pooling_fn = pooling_fn
        self.task_heads = ModuleDict({str(i): (
            Sequential(Linear(hidden_dim, head_hidden_dim), ReLU(), Linear(head_hidden_dim, out_dim)
                       ) if head_hidden_dim is not None
            else Linear(hidden_dim, out_dim)
        ) for i in tasks})

    def encode(self, data) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, data):
        x = self.encode(data)
        x = self.pooling_fn(x, data.batch)
        if self.dropout_p is not None:
            x = Fun.dropout(x, self.dropout_p, training=self.training)

        outs = []
        for i, r_target in enumerate(data.r_target):
            head = self.task_heads[str(int(r_target))]
            outs.append(head(x[i].unsqueeze(0)))

        return torch.cat(outs, dim=0).squeeze(-1)


class GCN(BaseMTLGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, tasks, head_hidden=None, dropout=None):
        super().__init__(tasks=tasks, hidden_dim=hidden_dim, out_dim=out_dim, head_hidden_dim=head_hidden, dropout_p=dropout)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class TransformerCN(BaseMTLGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, tasks, heads=4, concat=True, edge_dim=4, head_hidden=None,
                 dropout=None):
        emb_dim = hidden_dim * heads if concat else hidden_dim
        super().__init__(tasks=tasks, hidden_dim=emb_dim, out_dim=out_dim, head_hidden_dim=head_hidden, dropout_p=dropout)

        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)
        c1 = hidden_dim * heads if concat else hidden_dim
        self.conv2 = TransformerConv(c1, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)
        c2 = hidden_dim * heads if concat else hidden_dim
        self.conv3 = TransformerConv(c2, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)

    def encode(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        return x


class Gatv2CN(BaseMTLGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, tasks, heads=4, concat=True, edge_dim=4, head_hidden=None,
                 dropout=None):
        emb_dim = hidden_dim * heads if concat else hidden_dim
        super().__init__(tasks=tasks, hidden_dim=emb_dim, out_dim=out_dim, head_hidden_dim=head_hidden, dropout_p=dropout)

        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)
        c1 = hidden_dim * heads if concat else hidden_dim
        self.conv2 = GATv2Conv(c1, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)
        c2 = hidden_dim * heads if concat else hidden_dim
        self.conv3 = GATv2Conv(c2, hidden_dim, heads=heads, concat=concat, edge_dim=edge_dim)

    def encode(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr).relu()
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        return x


class GIN(BaseMTLGNN):
    def __init__(self, in_dim, hidden_dim, out_dim, tasks, head_hidden=None, dropout=None):
        super().__init__(tasks=tasks, hidden_dim=hidden_dim, out_dim=out_dim, head_hidden_dim=head_hidden, dropout_p=dropout)

        def make_mlp(in_dim_mlp, hidden_dim_mlp):
            return Sequential(
                Linear(in_dim_mlp, hidden_dim_mlp),
                BatchNorm1d(hidden_dim_mlp),
                ReLU(),
                Linear(hidden_dim_mlp, hidden_dim_mlp),
                ReLU()
            )

        self.conv1 = GINConv(make_mlp(in_dim, hidden_dim))
        self.conv2 = GINConv(make_mlp(hidden_dim, hidden_dim))
        self.conv3 = GINConv(make_mlp(hidden_dim, hidden_dim))

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x