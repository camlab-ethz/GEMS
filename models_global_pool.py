import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, Dropout
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, GINEConv
from torch_geometric.data import Batch


class GAT0bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0bn, self).__init__()

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
        return x


class GAT1bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT1bn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=1, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(256)
        self.conv2 = GATv2Conv(256, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
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
        return x


class GAT2bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT2bn, self).__init__()

        # Dimensionality reduction layers
        self.dim_reduction1 = Linear(in_channels, 256)
        self.dim_reduction2 = Linear(256, 64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = Dropout(dropout_prob)
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, graphbatch):



        # Proceed with graph convolutions
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
        self.dim_reduction1 = Linear(in_channels, 128)
        self.dim_reduction2 = Linear(128, 64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = Dropout(dropout_prob)
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        # Proceed with the graph convolutions
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
        self.dim_reduction1 = Linear(in_channels, 256)
        self.dim_reduction2 = Linear(256, 128)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = Dropout(dropout_prob)
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction
        x = self.dim_reduction1(graphbatch.x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        # Proceed with the graph convolutions
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
        self.norm = nn.LayerNorm(transformer_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.input_linear(x) # Project to transformer_dim
        x = self.norm(x) # LayerNorm
        attn_output, _ = self.attention(x, x, x) # Self-attention
        x = x + self.dropout(attn_output) # Dropout and residual connection
        x = self.norm(x) # LayerNorm
        x = x + self.dropout(self.feed_forward(x)) # Feed Forward Model, Dropout and residual connection
        x = self.norm(x) # LayerNorm
        x = self.output_linear(x) # Project to output_dim
        x = self.norm(x) # LayerNorm
        return x


class GAT5bn(nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bn, self).__init__()

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

        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




### This code could be changed to use MultiheadAttention from PyTorch

class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = Linear(in_channels, in_channels)
        self.key = Linear(in_channels, in_channels)
        self.value = Linear(in_channels, in_channels)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_scores = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5), dim=-1)
        attended_features = torch.matmul(attention_scores, V)
        
        return attended_features


class GAT6bn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT6bn, self).__init__()

        self.attention = SelfAttention(in_channels)  # Attention mechanism
        
        # Dimensionality reduction layers
        self.dim_reduction1 = Linear(in_channels, 256)
        self.dim_reduction2 = Linear(256, 64)

        # Convolutional Layers
        self.conv1 = GATv2Conv(64, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)

        self.dropout_layer = Dropout(dropout_prob)
        self.fc1 = Linear(256, 64)
        self.fc2 = Linear(64, 1)

    def forward(self, graphbatch):

        # Apply self-attention to input features
        x = self.attention(graphbatch.x)
        
        # Dimensionality reduction
        x = self.dim_reduction1(x)
        x = F.relu(x)
        x = self.dim_reduction2(x)

        # Proceed with original forward pass
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