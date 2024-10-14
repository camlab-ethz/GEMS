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
GATE9
BASED ON GATE3 but
- global feature is larger (n=256)
- global feature computation is based on global add pooling on nodes
- NodeModel is a GATEv2Conv layer 

architectures with 2-3 layers with prediction based on updating the global features of the model
in each layer of the model. A final regression head transforms the global vector into a prediction.

The architectures are as follows:

GATE9a: 2 layers, no residuals
GATE9b: 3 layers, no residuals
GATE9ar: 2 layers, with residuals
GATE9br: 3 layers, with residuals

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
        self.heads = 4

        self.conv = GATv2Conv(n_node_f, int(out_dim/self.heads), edge_dim=n_edge_f, heads=self.heads, dropout=dropout)

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = F.relu(self.conv(x, edge_index, edge_attr))
        if self.residuals:
            out = out + x
        return out


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_f, global_f, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + global_f, global_f), 
            nn.ReLU(), 
            nn.Linear(global_f, global_f))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)

#################################################################################################################
#################################################################################################################



class GATE9a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE9a, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(256)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        initial_global = torch.zeros((graphbatch.num_graphs, 256)).to(graphbatch.x.device)

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
    



class GATE9b(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE9b, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(256)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(256)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        initial_global = torch.zeros((graphbatch.num_graphs, 256)).to(graphbatch.x.device)

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
    



class GATE9ar(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE9ar, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(256)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        initial_global = torch.zeros((graphbatch.num_graphs, 256)).to(graphbatch.x.device)

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

        


class GATE9br(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE9br, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=256, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(256)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=True, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(256)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=256, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        initial_global = torch.zeros((graphbatch.num_graphs, 256)).to(graphbatch.x.device)

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

