import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as Fun


class GCN(torch.nn.Module):
    def __init__(self, dim_h,qm9_num_features, targets_num=1):
        super().__init__()
        self.conv1 = GCNConv(qm9_num_features, dim_h)
        self.conv2 = GCNConv(dim_h, dim_h)
        self.conv3 = GCNConv(dim_h, dim_h)
        self.lin = torch.nn.Linear(dim_h, targets_num)

    def forward(self, data):
        e = data.edge_index
        x = data.x

        x = self.conv1(x, e)
        x = x.relu()
        x = self.conv2(x, e)
        x = x.relu()
        x = self.conv3(x, e)

        x = global_mean_pool(x, data.batch)
        x = Fun.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x