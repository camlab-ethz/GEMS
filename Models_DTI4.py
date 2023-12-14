import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, GCNConv
from torch_geometric.data import Batch

    

'''BEST ARCHITECTURE SO FAR'''
class GAT0(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT0_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT0_mnp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        master_node_features = torch.cat((master_node_features, global_add_pool(x, batch=graphbatch.batch)), dim=1)

        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    



'''Just slightly more parameters'''
class GAT0a_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0a_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        master_node_features = torch.cat((master_node_features, global_add_pool(x, batch=graphbatch.batch)), dim=1)

        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



'''Much more parameters ------------------------------------------------------------------------------------------------------'''

class GAT0b(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0b, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(2048, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):

        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT0b_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0b_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(2048, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class GAT0b_mnp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0b_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(2048, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        master_node_features = torch.cat((master_node_features, global_add_pool(x, batch=graphbatch.batch)), dim=1)

        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x








'''3-LAYER ARCHITECTURE ------------------------------------------------------------------------------------------------------'''
class GAT0c_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT0c_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv3 = GATv2Conv(512, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        master_node_features = torch.cat((master_node_features, global_add_pool(x, batch=graphbatch.batch)), dim=1)


        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
