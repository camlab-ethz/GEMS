import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, GCNConv
from torch_geometric.data import Batch

    

#-------------------------------------------------------------------------------------------------------------
'''BASELINE ARCHITECTURE'''

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
#-------------------------------------------------------------------------------------------------------------

'''SECOND LAYER LARGER - The idea of this architecture is that not all input features are consensed to 256 in the second layer'''

class GAT1(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT1, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
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


class GAT1_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT1_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
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



#-------------------------------------------------------------------------------------------------------------

'''FIRST LAYER LARGER - The idea of this architecture is that not all input features are consensed to 256 in the FIRST layer'''

class GAT2(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT2, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(2048, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
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


class GAT2_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT2_mn, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        self.conv2 = GATv2Conv(2048, 64, edge_dim=edge_dim, heads=4, dropout=conv_dropout_prob)
        
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
    

''' ------------------------------------------------------------------------------------------------------
FIRST LAYER AND SECOND LAYER LARGER '''

class GAT3(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT3, self).__init__()

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


class GAT3_mn(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim, conv_dropout_prob):
        super(GAT3_mn, self).__init__()

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






'''3-LAYER ARCHITECTURE ------------------------------------------------------------------------------------------------------'''

class GAT4_mn(torch.nn.Module):
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


        x = self.dropout_layer(master_node_features)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
