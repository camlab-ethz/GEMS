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
GATE16
BASED ON GATE3 but
- Initial node transformation with NodeTransform MLP
- global feature COMPUTATION IS FLEXIBLE (Can become larger)
- global feature is initialized with ChemBERTa embeddings
- global feature computation is based on global add pooling on nodes
- NodeModel is a GATEv2Conv layer 

architectures with 2 layers and no residuals with prediction based on updating the global features of the model
in each layer of the model. A final regression head transforms the global vector into a prediction.

The architectures are as follows:

GATE16a: Global feature is becoming larger up to 512
GATE16b: Initial Node and Global transformation with dropout 0.1


dropout and conv_dropout are possible
'''

class FeatureTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim, dropout):
        super(FeatureTransformMLP, self).__init__()
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


class GlobalModel(torch.nn.Module):
    def __init__(self, n_node_f, glob_f_in, glob_f_hidden, glob_f_out, dropout):
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)
        self.global_mlp = nn.Sequential(
            nn.Linear(n_node_f + glob_f_in, glob_f_hidden), 
            nn.ReLU(), 
            nn.Linear(glob_f_hidden, glob_f_out))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, global_add_pool(x, batch=batch)], dim=1)
        out = self.dropout_layer(out)
        return self.global_mlp(out)

#################################################################################################################
#################################################################################################################



class GATE16a(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE16a, self).__init__()

        self.NodeTransform = FeatureTransformMLP(in_channels, 256, 256, dropout=dropout_prob)
        
        self.layer1 = self.build_layer( node_f=256, node_f_hidden=256, node_f_out=256, 
                                        edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128,
                                        glob_f=384, glob_f_hidden=512, glob_f_out=512,
                                        residuals=False, dropout=conv_dropout_prob
                                        )
        
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)
        self.u_bn1 = BatchNorm1d(512)

        self.layer2 = self.build_layer( node_f=256, node_f_hidden=256, node_f_out=256,
                                        edge_f=128, edge_f_hidden=128, edge_f_out=128,
                                        glob_f=512, glob_f_hidden=512, glob_f_out=512,
                                        residuals=False, dropout=conv_dropout_prob
                                        )

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 1)

    def build_layer(self, 
                    node_f, node_f_hidden, node_f_out, 
                    edge_f, edge_f_hidden, edge_f_out,
                    glob_f, glob_f_hidden, glob_f_out,
                    residuals, dropout):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals, dropout=dropout),
            node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals, dropout=dropout),
            global_model=GlobalModel(node_f_out, glob_f, glob_f_hidden, glob_f_out, dropout=dropout)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index
        
        x = self.NodeTransform(graphbatch.x)

        x, edge_attr, u = self.layer1(x, edge_index, graphbatch.edge_attr, u=graphbatch.lig_emb, batch=graphbatch.batch)
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


