import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter, scatter_mean



'''
GATE11
BASED ON GATE0 but
- Initial node transformation with NodeTransform MLP
- Node Model is a GATv2Conv layer

architectures with 2-3 layers followed by a global add pooling (on nodes) 
and potentially with residual connections.

GATE11a: 2 layers, no residuals
GATE11b: 3 layers, no residuals
GATE11ar: 2 layers, with residuals
GATE11br: 3 layers, with residuals

dropout and conv_dropout are possible
'''

class NodeTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super(NodeTransformMLP, self).__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        x = self.mlp(node_features)
        return self.dropout_layer(x)


class EdgeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super().__init__()
        self.residuals = residuals
        self.dropout_layer = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * n_node_f + n_edge_f, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        out = self.dropout_layer(out)
        out = self.edge_mlp(out)
        if self.residuals:
            out = out + edge_attr
        return out


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.heads = 4

        self.conv = GATv2Conv(n_node_f, int(out_dim/self.heads), edge_dim=n_edge_f, heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(self.conv(x, edge_index, edge_attr))
        if self.residuals:
            out = out + x
        return out


#################################################################################################################
#################################################################################################################



class GATE11a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE11a, self).__init__()

        self.NodeTransform = NodeTransformMLP(in_channels, 256, 256, dropout_prob)

        # Build each layer separately
        self.layer1 = self.build_layer(node_f=256, edge_f=edge_dim, node_f_hidden=256, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        x = self.NodeTransform(graphbatch.x)

        x, edge_attr, _ = self.layer1(x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        x, _, _ = self.layer2(x, edge_index, edge_attr, None, batch=graphbatch.batch)

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    


class GATE11b(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE11b, self).__init__()

        self.NodeTransform = NodeTransformMLP(in_channels, 256, 256, dropout_prob)

        self.layer1 = self.build_layer(node_f=256, edge_f=edge_dim, node_f_hidden=256, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.edge_bn2 = BatchNorm1d(128)
        self.node_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        x = self.NodeTransform(graphbatch.x)

        x, edge_attr, _ = self.layer1(x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        x, edge_attr, _ = self.layer2(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)
        x = self.node_bn2(x)
        edge_attr = self.edge_bn2(edge_attr)

        x, _, _ = self.layer3(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out