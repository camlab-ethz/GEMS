import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import torch_geometric.nn as geom_nn
from torch_geometric.nn import GATv2Conv, GINConv, GINEConv
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


# class NodeModel(torch.nn.Module):
#     def __init__(self, n_node_f, n_edge_f, hidden_dim, out_dim, residuals):
#         super(NodeModel, self).__init__()
#         self.residuals = residuals
#         self.node_mlp_1 = nn.Sequential(
#             nn.Linear(n_node_f + n_edge_f, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )
#         self.node_mlp_2 = nn.Sequential(
#             nn.Linear(hidden_dim + n_node_f, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim),
#         )

#     def forward(self, x, edge_index, edge_attr, u, batch):
#         row, col = edge_index
#         out = torch.cat([x[col], edge_attr], dim=1) # Concatenate destination node features with edge features
#         out = self.node_mlp_1(out) # Apply first MLP to the concatenated edge features
#         out = scatter_mean(out, row, dim=0, dim_size=x.size(0)) # Map edge features back to source nodes
#         out = torch.cat([x, out], dim=1)
#         out = self.node_mlp_2(out)
#         if self.residuals:
#             out = out + x
#         return out


# Global Model pools edge features and returns a transformed edge representation
class GlobalModel(torch.nn.Module):
    def __init__(self, n_edge_f, global_f):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(n_edge_f + global_f, int(n_edge_f/2)), 
            nn.ReLU(), 
            nn.Linear(int(n_edge_f/2), global_f))

    def forward(self, x, edge_index, edge_attr, u, batch):
        out = torch.cat([u, scatter(edge_attr, batch[edge_index[0]], dim=0, reduce='mean')], dim=1)
        return self.global_mlp(out)

#################################################################################################################
#################################################################################################################


# Same as GATEh5bn but with residuals
class GATEh8bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEh8bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        
        GIN_nn1 = nn.Linear(in_channels, 256)
        self.conv1 = GINEConv(GIN_nn1, train_eps=False, edge_dim=128)
        
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)

        GIN_nn2 = nn.Linear(256, 512)
        self.conv2 = GINEConv(GIN_nn2, train_eps=False, edge_dim=256)
        
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=True)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index, edge_attr)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, edge_attr)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u






# Same as GATEg5bn but with residuals
class GATEg8bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEg8bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        
        GIN_nn1 = nn.Linear(in_channels, 256)
        self.conv1 = GINConv(GIN_nn1, train_eps=False)
        
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)

        GIN_nn2 = nn.Linear(256, 512)
        self.conv2 = GINConv(GIN_nn2, train_eps=False)
        
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=True)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u






# Same as GATEv5bn but with residuals
class GATEv8bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEv8bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)
        self.conv2 = GATv2Conv(256, 128, edge_dim=256, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=True)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index, edge_attr)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, edge_attr)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u






# Regression is performed by the global model based on EDGE features (with three layers)
class GATEh5bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEh5bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        
        GIN_nn1 = nn.Linear(in_channels, 256)
        self.conv1 = GINEConv(GIN_nn1, train_eps=False, edge_dim=128)
        
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)

        GIN_nn2 = nn.Linear(256, 512)
        self.conv2 = GINEConv(GIN_nn2, train_eps=False, edge_dim=256)
        
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=False)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index, edge_attr)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, edge_attr)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u




# Regression is performed by the global model based on EDGE features (with three layers)
class GATEg5bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEg5bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        
        GIN_nn1 = nn.Linear(in_channels, 256)
        self.conv1 = GINConv(GIN_nn1, train_eps=False)
        
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)

        GIN_nn2 = nn.Linear(256, 512)
        self.conv2 = GINConv(GIN_nn2, train_eps=False)
        
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=False)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u






# Regression is performed by the global model based on EDGE features (with three layers)
class GATEv5bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATEv5bn, self).__init__()
        
        # Build each layer separately
        self.layer1 = self.build_layer(node_f=in_channels, edge_f=edge_dim, edge_f_hidden=64, edge_f_out=128, residuals=False)
        self.conv1 = GATv2Conv(in_channels, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(256)
        self.edge_bn1 = BatchNorm1d(128)

        self.layer2 = self.build_layer(node_f=256, edge_f=128, edge_f_hidden=128, edge_f_out=256, residuals=False)
        self.conv2 = GATv2Conv(256, 128, edge_dim=256, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(512)
        self.edge_bn2 = BatchNorm1d(256)

        self.layer3 = self.build_layer(node_f=512, edge_f=256, edge_f_hidden=256, edge_f_out=256, residuals=False)
        
        # self.dropout_layer = nn.Dropout(dropout_prob)
        # self.fc1 = nn.Linear(256, 64)
        # self.fc2 = nn.Linear(64, 1)

    def build_layer(self, node_f, edge_f, edge_f_hidden, edge_f_out, residuals):
        return geom_nn.MetaLayer(
            edge_model=EdgeModel(node_f, edge_f, edge_f_hidden, edge_f_out, residuals=residuals),
            #node_model=NodeModel(node_f, edge_f_out, node_f_hidden, node_f_out, residuals=residuals),
            global_model=GlobalModel(edge_f_out, global_f=1)
        )

    def forward(self, graphbatch):
        edge_index = graphbatch.edge_index

        # Initialize global feature of length 1 for each graph in the batch
        initial_global = torch.zeros((graphbatch.num_graphs, 1)).to(graphbatch.x.device)

        # Update edge features and global feature
        _, edge_attr, u = self.layer1(graphbatch.x, edge_index, graphbatch.edge_attr, u=initial_global, batch=graphbatch.batch)
        edge_attr = self.edge_bn1(edge_attr)

        # Update node features
        x = self.node_bn1(F.relu(self.conv1(graphbatch.x, edge_index, edge_attr)))

        # Update edge features and global features
        _, edge_attr, u = self.layer2(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        edge_attr = self.edge_bn2(edge_attr)

        # Updata node features
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, edge_attr)))

        # Update edge features and global features
        _, _, u = self.layer3(x, edge_index, edge_attr, u, batch=graphbatch.batch)
        
        return u
