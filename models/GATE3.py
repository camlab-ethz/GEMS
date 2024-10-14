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
GATE3 are architectures with 2-3 layers with prediction based on updating the global features (n=64) of the model
in each layer of the model. A final regression head transforms the global vector into a prediction.

The architectures are as follows:

GATE3a: 2 layers, no residuals
GATE3b: 3 layers, no residuals
GATE3ar: 2 layers, with residuals
GATE3br: 3 layers, with residuals

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


class NodeModel(torch.nn.Module):
    def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals, dropout):
        super(NodeModel, self).__init__()
        self.residuals = residuals
        self.dropout_layer = nn.Dropout(dropout)
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
        out = self.dropout_layer(out)
        out = torch.cat([x, out], dim=1) # Concatenate source node features with aggregated edge features
        out = self.node_mlp_2(out) # Apply second MLP
        if self.residuals:
            out = out + x
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



class GATE3a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE3a, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(64)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(edge_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 64)).to(graphbatch.x.device)

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        _, _, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        u = self.dropout_layer(u)

        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    



class GATE3ar(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE3ar, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(64)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(edge_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 64)).to(graphbatch.x.device)

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        _, _, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        u = self.dropout_layer(u)

        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)
        return out




class GATE3b(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE3b, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(64)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(64)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(edge_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 64)).to(graphbatch.x.device)

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        x, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        x = self.node_bn2(x)
        edge_attr = self.edge_bn2(edge_attr)
        u = self.u_bn2(u)
        
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        u = self.dropout_layer(u)
        
        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    



class GATE3br(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE3br, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=64, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(64)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=True, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(64)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=64, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(edge_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 64)).to(graphbatch.x.device)

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        x = self.node_bn1(x)
        edge_attr = self.edge_bn1(edge_attr)
        u = self.u_bn1(u)

        x, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        x = self.node_bn2(x)
        edge_attr = self.edge_bn2(edge_attr)
        u = self.u_bn2(u)
        
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        u = self.dropout_layer(u)
        
        # Fully-Connected Layers
        out = self.fc1(u)
        out = F.relu(out)
        out = self.fc2(out)
        return out

