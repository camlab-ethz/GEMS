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
GATE14
BASED ON GATE3 but
- global feature is larger (n=256)
- global feature is initialized with ChemBERTa embeddings
- global feature computation is based on global add pooling on nodes

architectures with 2-3 layers with prediction based on updating the global features of the model
in each layer of the model. A final regression head transforms the global vector into a prediction.

The architectures are as follows:

GATE14a: 2 layers, no residuals
GATE14b: 3 layers, no residuals
GATE14ar: 2 layers, with residuals
GATE14br: 3 layers, with residuals

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



class GATE14a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE14a, self).__init__()
        
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
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
    



class GATE14ar(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE14ar, self).__init__()
        
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
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




class GATE14b(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE14b, self).__init__()
        
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(384)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
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
    



class GATE14br(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE14br, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, node_f_hidden=128, edge_f_hidden=64, node_f_out=256, edge_f_out=128, global_f=384, residuals=False, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(384)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=True, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_bn2 = BatchNorm1d(128)
        self.u_bn2 = BatchNorm1d(384)
        
        self.layer3 = self.build_layer(node_f=256, edge_f=128, node_f_hidden=256, edge_f_hidden=128, node_f_out=256, edge_f_out=128, global_f=384, residuals=True, dropout=conv_dropout_prob)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, node_f_hidden, edge_f_hidden, node_f_out, edge_f_out, global_f, residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, global_f, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        x, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
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

