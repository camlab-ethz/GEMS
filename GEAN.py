import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATv2Conv, global_add_pool


class EdgeTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, out_dim):
        super(EdgeTransformMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim + 2 * node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, edge_index, node_features, edge_features):
        src_node_features = node_features[edge_index[0]]
        tgt_node_features = node_features[edge_index[1]]
        concatenated_features = torch.cat([edge_features, src_node_features, tgt_node_features], dim=1)
        return self.mlp(concatenated_features)


class NodeTransformMLP(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, out_dim):
        super(NodeTransformMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim))

    def forward(self, node_features):
        return self.mlp(node_features)


class GEAN_template(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GEAN_template, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.1)
        self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(xe_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out




# This first class of GEAN should be identical to GAT*bn architectures, but with edge updates added
class GAT4bnE(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT4bnE, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        #self.node_dropout = nn.Dropout(0.1)
        #self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        #x = self.node_dropout(x)
        #e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class GAT5bnE(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bnE, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.1)
        self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class GAT6bnE(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT6bnE, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=512, out_dim=256)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.1)
        self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(256, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



class GAT5bn2E(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bn2E, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.2)
        self.edge_dropout = nn.Dropout(0.2)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class GAT6bn2E(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT6bn2E, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=512, out_dim=256)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.2)
        self.edge_dropout = nn.Dropout(0.2)

        # Convolutional Layers
        self.conv1 = GATv2Conv(256, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out




## In this use LayerNorm instead of BatchNorm
class GAT6lnE(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT6lnE, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=512, out_dim=256)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.1)
        self.edge_dropout = nn.Dropout(0.1)

        # Convolutional Layers
        self.conv1 = GATv2Conv(256, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.ln1 = nn.LayerNorm(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.ln2 = nn.LayerNorm(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.ln1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_transform2(edge_index, x, e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.ln2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out




## In this class I also normalize the edge features
class GAT5bn2Ebn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bn2Ebn, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.2)
        self.edge_dropout = nn.Dropout(0.2)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)
        self.edge_bn1 = BatchNorm1d(128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_bn1(self.edge_transform2(edge_index, x, e))
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class GAT6bn2Ebn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT6bn2Ebn, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=512, out_dim=256)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.2)
        self.edge_dropout = nn.Dropout(0.2)

        # Convolutional Layers
        self.conv1 = GATv2Conv(256, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)
        self.edge_bn1 = BatchNorm1d(128)

        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        #self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_bn1(self.edge_transform2(edge_index, x, e))
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, e)))
        #e = self.edge_transform3(edge_index, x, e)
        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        #e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        #xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(x_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

















# This third class of GEAN should be identical to GAT*bn2E architectures, but the updated edge features should flow into the 
# final regression head of the model together with the node features

class GAT5bn2Exe(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT5bn2Exe, self).__init__()

        # Transform input features
        self.node_transform = NodeTransformMLP(in_channels, hidden_dim=256, out_dim=128)
        self.edge_transform1 = EdgeTransformMLP(node_feature_dim=128, edge_feature_dim=edge_dim, hidden_dim=64, out_dim=64)
        self.node_dropout = nn.Dropout(0.2)
        self.edge_dropout = nn.Dropout(0.2)

        # Convolutional Layers
        self.conv1 = GATv2Conv(128, 256, edge_dim=64, heads=4, dropout=conv_dropout_prob)
        self.node_bn1 = BatchNorm1d(1024)
        self.edge_transform2 = EdgeTransformMLP(node_feature_dim=1024, edge_feature_dim=64, hidden_dim=128, out_dim=128)
        self.edge_bn1 = BatchNorm1d(128)


        self.conv2 = GATv2Conv(1024, 64, edge_dim=128, heads=4, dropout=conv_dropout_prob)
        self.node_bn2 = BatchNorm1d(256)
        self.edge_transform3 = EdgeTransformMLP(node_feature_dim=256, edge_feature_dim=128, hidden_dim=256, out_dim=256)
        self.edge_bn2 = BatchNorm1d(256)

        self.dropout_layer = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, graphbatch):

        # Dimensionality reduction of node features, then update edge_features
        edge_index = graphbatch.edge_index
        x = self.node_transform(graphbatch.x)
        e = self.edge_transform1(edge_index, x, graphbatch.edge_attr)
        
        x = self.node_dropout(x)
        e = self.edge_dropout(e)
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn1(F.relu(self.conv1(x, edge_index, e)))
        e = self.edge_bn1(self.edge_transform2(edge_index, x, e))
        
        # Update the node features (x) with convolution, then update the edge_features (e)
        x = self.node_bn2(F.relu(self.conv2(x, edge_index, e)))
        e = self.edge_bn2(self.edge_transform3(edge_index, x, e))

        
        # Pool the node features (x) and edge_features (e) of each interaction graph
        x_pool = global_add_pool(x, batch=graphbatch.batch)
        e_pool = global_add_pool(e, batch=graphbatch.batch[edge_index[0]])
        xe_pool = torch.cat([x_pool, e_pool], dim=1)
        
        # Fully-Connected Layers
        out = self.dropout_layer(xe_pool)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out