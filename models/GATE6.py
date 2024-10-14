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
GATE6
BASED ON GATE0 
- Node Model is replaced with a GATv2Conv layer

are architectures with 2-3 layers followed by a global add pooling (on nodes) 
and potentially with residual connections.

GATE6a: 2 layers, no residuals
GATE6b: 3 layers, no residuals
GATE6ar: 2 layers, with residuals
GATE6br: 3 layers, with residuals

dropout and conv_dropout are possible
'''




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


# Global Model pools edge features and returns a transformed edge representation
class GlobalModel(torch.nn.Module):
    def __init__(self, n_edge_f, global_f, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_edge_f + global_f, int(n_edge_f/2)), 
            nn.ReLU(), 
            nn.Linear(int(n_edge_f/2), global_f))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, scatter(edge_attr, batch[edge_index[0]], dim=0, reduce='mean')], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)

#################################################################################################################
#################################################################################################################



class GATE6a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE6a, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            #global_model=GlobalModel(node_f_out, global_f=1, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        _, edge_attr, _ = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv1(graphbatch.x, edge_index, edge_attr))
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        _, edge_attr, _ = self.layer2(x, edge_index, edge_attr, None, batch=graphbatch.batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    


class GATE6b(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE6b, self).__init__()
     
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.edge_bn2 = BatchNorm1d(128)
        self.node_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv3 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            #global_model=GlobalModel(node_f_out, global_f=1, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        _, edge_attr, _ = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv1(graphbatch.x, edge_index, edge_attr))
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        _, edge_attr, _ = self.layer2(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.node_bn2(x)
        edge_attr = self.edge_bn2(edge_attr)

        _, edge_attr, _ = self.layer3(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    



class GATE6ar(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE6ar, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=True, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            #global_model=GlobalModel(node_f_out, global_f=1, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        _, edge_attr, _ = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv1(graphbatch.x, edge_index, edge_attr))
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        _, edge_attr, _ = self.layer2(x, edge_index, edge_attr, None, batch=graphbatch.batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    




class GATE6br(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE6br, self).__init__()
     
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, residuals=False, dropout=conv_dropout_prob)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=True, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.edge_bn2 = BatchNorm1d(128)
        self.node_bn2 = BatchNorm1d(256)
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, residuals=True, dropout=conv_dropout_prob)
        self.conv3 = GATv2Conv(256, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            #global_model=GlobalModel(node_f_out, global_f=1, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        _, edge_attr, _ = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv1(graphbatch.x, edge_index, edge_attr))
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)

        _, edge_attr, _ = self.layer2(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.node_bn2(x)
        edge_attr = self.edge_bn2(edge_attr)

        _, edge_attr, _ = self.layer3(x, edge_index, edge_attr, u=None, batch=graphbatch.batch)
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        # Pool the nodes of each interaction graph
        out = global_add_pool(x, graphbatch.batch)
        out = self.dropout_layer(out)
        
        # Fully-Connected Layers
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out