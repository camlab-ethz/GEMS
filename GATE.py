import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_add_pool
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops



class EdgeUpdateConv(MessagePassing):
    def __init__(self, node_in_channels, edge_in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation.
        self.lin_node = nn.Linear(node_in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_node.reset_parameters()
        self.lin_edge.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, node_in_channels]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, edge_in_channels]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))

        # Step 2: Linearly transform node and edge feature matrices.
        x_transformed = self.lin_node(x)
        edge_attr = self.lin_edge(edge_attr)

        # Step 3-4: Start propagating messages to update edge features.
        updated_edge_attr = self.propagate(edge_index, x=x_transformed, edge_attr=edge_attr)

        # Step 5: Apply a final bias vector to edge features.
        updated_edge_attr = updated_edge_attr + self.bias

        # Node features remain unchanged, only updated edge features are returned.
        return updated_edge_attr

    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, out_channels] - features of target nodes
        # x_j has shape [E, out_channels] - features of source nodes
        # edge_attr has shape [E, out_channels] - edge features

        # Combine source node features, target node features, and edge features.
        return x_i + x_j + edge_attr



class NodeTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim):
        super(NodeTransformMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, node_features):
        return self.mlp(node_features)


class EdgeTransformMLP(nn.Module):
    def __init__(self, edge_feature_dim, hidden_dim, out_dim):
        super(EdgeTransformMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )

    def forward(self, edge_features):
        return self.mlp(edge_features)



class GATE(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATE, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=512, out_dim=256)
        self.edge_transform = EdgeTransformMLP(edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.1)
        self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(256, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(1024)
        self.edge_update1 = EdgeUpdateConv(node_in_channels=1024, edge_in_channels=64, out_channels=128)
        self.edge_bn1 = BatchNorm1d(128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_update2 = EdgeUpdateConv(node_in_channels=256, edge_in_channels=128, out_channels=256)
        self.edge_bn2 = BatchNorm1d(256)

        # Final regression based on edges
        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Transform node features and edge features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform(graphbatch.edge_attr)
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Convolutional Layers
        x = self.node_bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_bn1(F.relu(self.edge_update1(x, edge_index, e)))
        
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, e)))
        e = self.edge_bn2(F.relu(self.edge_update2(x, edge_index, e)))
        

        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out