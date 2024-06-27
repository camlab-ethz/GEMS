import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_add_pool, GINEConv
from torch_geometric.data import Batch


class GAT0bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0bn, self).__init__()

        # Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT1bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT1bn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=1, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(256)
        self.conv2 = GATv2Conv(256, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class GAT2bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT2bn, self).__init__()

        # Dimensionality reduction layers
        self.dim_reduction1 = nn.Linear(in_channels, 256)
        self.dim_reduction2 = nn.Linear(256, 64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        x = self.conv1(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




class GAT3bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT3bn, self).__init__()

        # Dimensionality reduction layers
        self.dim_reduction1 = nn.Linear(in_channels, 128)
        self.dim_reduction2 = nn.Linear(128, 64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        x = self.conv1(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class GAT4bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT4bn, self).__init__()

        # Dimensionality reduction layers
        self.dim_reduction1 = nn.Linear(in_channels, 256)
        self.dim_reduction2 = nn.Linear(256, 128)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        x = self.conv1(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT5bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bn, self).__init__()

        # Dimensionality reduction layers
        self.dim_reduction1 = nn.Linear(in_channels, 256)
        self.dim_reduction2 = nn.Linear(256, 128)
        self.dropout_layer0 = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)
        x = self.dropout_layer0(x)

        x = self.conv1(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x










class EdgeUpdateMLP(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, out_dim):
        super(EdgeUpdateMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim + 2 * node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, edge_index, node_features, edge_features):

        src_node_features = node_features[edge_index[0]]
        tgt_node_features = node_features[edge_index[1]]
        concatenated_features = torch.cat([edge_features, src_node_features, tgt_node_features], dim=1)
        updated_edge_features = self.mlp(concatenated_features)
        
        return updated_edge_features
    

class NodeUpdateMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim):
        super(NodeUpdateMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, node_features):
        return self.mlp(node_features)



class GEAT(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GEAT, self).__init__()

        # Transform input features
        self.dim_reduction = NodeUpdateMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_update1 = EdgeUpdateMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_update2 = EdgeUpdateMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_update3 = EdgeUpdateMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.dim_reduction(graphbatch.x)
        edge_attr = self.edge_update1(edge_index, x, graphbatch.edge_attr)

        # Update the node features with convolution, then update the edge_features
        x = self.bn1(F.relu(self.conv1(x, edge_index, edge_attr)))
        edge_attr = self.edge_update2(edge_index, x, edge_attr)

        # Update the node features with convolution, then update the edge_features
        x = self.bn2(F.relu(self.conv2(x, edge_index, edge_attr)))
        edge_attr = self.edge_update3(edge_index, x, edge_attr)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        #edge_attr = global_add_pool(edge_attr, batch=graphbatch.batch[edge_index[0]])
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x






####################################################################################################################

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, transformer_dim, output_dim, num_heads, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=transformer_dim, num_heads=num_heads, dropout=dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(transformer_dim, 4 * transformer_dim),  # Expand dimensions
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * transformer_dim, transformer_dim)  # Compress back to model_dim
        )
        
        self.input_linear = nn.Linear(input_dim, transformer_dim)
        self.output_linear = nn.Linear(transformer_dim, output_dim)
        self.norm1 = nn.LayerNorm(transformer_dim)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.input_linear(x) # Project to transformer_dim
        x = self.norm1(x) # LayerNorm
        attn_output, _ = self.attention(x, x, x) # Self-attention
        x = x + self.dropout(attn_output) # Dropout and residual connection
        x = self.norm1(x) # LayerNorm
        x = x + self.dropout(self.feed_forward(x)) # Feed Forward Model, Dropout and residual connection
        x = self.norm1(x) # LayerNorm
        x = self.output_linear(x) # Project to output_dim
        x = self.norm2(x) # LayerNorm
        return x


class GATTbn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GATTbn, self).__init__()

        self.transformer = TransformerBlock(input_dim=in_channels, transformer_dim=128, output_dim=64, num_heads=4)
        
        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = nn.BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = graphbatch.x
        x = self.transformer(x)  # Apply transformer block for dimensionality reduction
        
        x = self.conv1(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x





































































# ______________________________________________________________________________________________________


# Same as GAT0bn but with final ReLu
class GAT0bn1(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0bn1, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x
    


# With Dropout layer between the two fully-connected layers
class GAT0bn2(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0bn2, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x


class GIN0(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GIN0, self).__init__()
        
        fc_gin1 = Sequential(Linear(in_channels, 256), ReLU(), Linear(256, 256))
        self.conv1 = GINEConv(fc_gin1, edge_dim=edge_dim)
        self.bn1 = BatchNorm1d(256)

        fc_gin2 = Sequential(Linear(256, 128), ReLU(), Linear(128, 128))
        self.conv2 = GINEConv(fc_gin2,edge_dim=edge_dim)
        self.bn2 = BatchNorm1d(128)
        
        fc_gin3 = Sequential(Linear(128, 64), ReLU(), Linear(64, 64))
        self.conv3 = GINEConv(fc_gin3, edge_dim=edge_dim)
        self.bn3 = BatchNorm1d(64)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)

        self.fc1 = torch.nn.Linear(64, 16)
        self.fc2 = torch.nn.Linear(16, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.bn3(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x