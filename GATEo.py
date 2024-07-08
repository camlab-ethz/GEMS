import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter, scatter_mean



class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals):
        super().__init__()
        self.residuals = residuals
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(hidden_dim + n_node_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[col], edge_attr], dim=1) # Concatenate destination node features with edge features
        out = self.node_mlp_1(out) # Apply first MLP to the concatenated edge features
        out = scatter_mean(out, row, dim=0, dim_size=x.size(0)) # Map edge features back to source nodes
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)
        if self.residuals:
            out = out + x
        return out


# Global Model pools edge features and returns a transformed edge representation
class GlobalModel(torch.nn.Module):
    def __init__(self, n_edge_f, global_f):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(n_edge_f + global_f, int(n_edge_f/2)), 
            nn.ReLU(), 
            nn.Linear(int(n_edge_f/2), global_f))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = scatter(edge_attr, batch[edge_index[0]], dim=0, reduce='mean')
        return self.global_mlp(out)

#################################################################################################################
#################################################################################################################


