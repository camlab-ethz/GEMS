import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, GCNConv


'''
# MODEL ARCHITECTURE SMALL BASELINE WITH GLOBAL MEAN POOL 
'''
class GATbaseline_mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GATbaseline_mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        # First convolution only between ligand and protein nodes
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)
        # Fully-Connected Layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


'''
# MODEL ARCHITECTURE LARGE BASELINE WITH GLOBAL MEAN POOL 
'''

class GATbaseline1_mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GATbaseline1_mp, self).__init__()
        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 640, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(640, 256, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(256, 128, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(128, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        # First convolution only between ligand and protein nodes
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x








# '''
# # MODEL GATN2 with multiple attention heads
# '''

# class GATN_transformer2(torch.nn.Module):
#     def __init__(self, dropout_prob, in_channels, edge_dim):
#         super(GATN_transformer2, self).__init__()

#         #Convolutional Layers
#         self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim, heads=8)
#         self.conv2 = GATv2Conv(in_channels*8, 1024, edge_dim=edge_dim)
#         self.conv3 = GATv2Conv(1024, 512, edge_dim=edge_dim)
#         self.conv4 = GATv2Conv(512, 256, edge_dim=edge_dim)

#         self.dropout_layer = torch.nn.Dropout(dropout_prob)

#         self.fc1 = torch.nn.Linear(256, 64)
#         self.bn1 = torch.nn.BatchNorm1d(64)
#         self.fc2 = torch.nn.Linear(64, 1)


#     #def forward(self, x, edge_index_prot, edge_attr_prot, edge_index_lig, edge_attr_lig, batch):
#     def forward(self, graphbatch):
        
#         # First convolution only between ligand and protein nodes
#         x = self.conv1(graphbatch.x, graphbatch.edge_index_prot, graphbatch.edge_attr_prot)
#         x = F.relu(x)

#         # Remaining convolution only within ligand
#         x = self.conv2(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv3(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv4(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)

#         # Pool the nodes of each interaction graph
#         x = global_add_pool(x, batch=graphbatch.batch)

#         # Dropout
#         x = self.dropout_layer(x)

#         # Fully-Connected Layers
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x



# '''
# # MODEL GATN with multiple attention heads
# '''

# class GATN_transformer(torch.nn.Module):
#     def __init__(self, dropout_prob, in_channels, edge_dim):
#         super(GATN_transformer, self).__init__()

#         #Convolutional Layers
#         self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim, heads=5)
#         self.conv2 = GATv2Conv(in_channels*5, 1024, edge_dim=edge_dim)
#         self.conv3 = GATv2Conv(1024, 512, edge_dim=edge_dim)
#         self.conv4 = GATv2Conv(512, 256, edge_dim=edge_dim)

#         self.dropout_layer = torch.nn.Dropout(dropout_prob)

#         self.fc1 = torch.nn.Linear(256, 64)
#         self.bn1 = torch.nn.BatchNorm1d(64)
#         self.fc2 = torch.nn.Linear(64, 1)


#     #def forward(self, x, edge_index_prot, edge_attr_prot, edge_index_lig, edge_attr_lig, batch):
#     def forward(self, graphbatch):
        
#         # First convolution only between ligand and protein nodes
#         x = self.conv1(graphbatch.x, graphbatch.edge_index_prot, graphbatch.edge_attr_prot)
#         x = F.relu(x)

#         # Remaining convolution only within ligand
#         x = self.conv2(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv3(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv4(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)

#         # Pool the nodes of each interaction graph
#         x = global_add_pool(x, batch=graphbatch.batch)

#         # Dropout
#         x = self.dropout_layer(x)

#         # Fully-Connected Layers
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x




# '''
# # MODEL ARCHITECTURE 1 
# '''

# class GATNnorm(torch.nn.Module):
#     def __init__(self, dropout_prob, in_channels, edge_dim):
#         super(GATNnorm, self).__init__()

#         #Convolutional Layers
#         self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim)
#         self.conv2 = GATv2Conv(in_channels, 512, edge_dim=edge_dim)
#         self.conv3 = GATv2Conv(512, 1024, edge_dim=edge_dim)
#         self.conv4 = GATv2Conv(1024, 256, edge_dim=edge_dim)

#         self.dropout_layer = torch.nn.Dropout(dropout_prob)

#         self.fc1 = torch.nn.Linear(256, 64)
#         self.bn1 = torch.nn.BatchNorm1d(64)
#         self.fc2 = torch.nn.Linear(64, 1)


#     #def forward(self, x, edge_index_prot, edge_attr_prot, edge_index_lig, edge_attr_lig, batch):
#     def forward(self, graphbatch):
        
#         # First convolution only between ligand and protein nodes
#         x = self.conv1(graphbatch.x, graphbatch.edge_index_prot, graphbatch.edge_attr_prot)
#         x = F.relu(x)

#         # Remaining convolution only within ligand
#         x = self.conv2(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv3(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)
#         x = self.conv4(x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
#         x = F.relu(x)

#         # Pool the nodes of each interaction graph
#         x = global_add_pool(x, batch=graphbatch.batch)

#         # Dropout
#         x = self.dropout_layer(x)

#         # Fully-Connected Layers
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x



# '''
# # MODEL ARCHITECTURE 1 
# '''

# class GATNbase3(torch.nn.Module):
#     def __init__(self, dropout_prob, in_channels, edge_dim):
#         super(GATNbase3, self).__init__()

#         #Convolutional Layers
#         self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim)
#         self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim)
#         self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)

#         self.dropout_layer = torch.nn.Dropout(dropout_prob)

#         self.fc1 = torch.nn.Linear(256, 64)
#         self.bn1 = torch.nn.BatchNorm1d(64)
#         self.fc2 = torch.nn.Linear(64, 1)


#     #def forward(self, x, edge_index_prot, edge_attr_prot, edge_index_lig, edge_attr_lig, batch):
#     def forward(self, graphbatch):
        
#         # First convolution only between ligand and protein nodes
#         x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)
#         x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)

#         # Pool the nodes of each interaction graph
#         x = global_add_pool(x, batch=graphbatch.batch)

#         # Dropout
#         x = self.dropout_layer(x)

#         # Fully-Connected Layers
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x




# '''
# # MODEL ARCHITECTURE 1 
# '''

# class GATNbase3t(torch.nn.Module):
#     def __init__(self, dropout_prob, in_channels, edge_dim):
#         super(GATNbase3t, self).__init__()

#         #Convolutional Layers
#         self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=2)
#         self.conv2 = GATv2Conv(512*2, 1024, edge_dim=edge_dim, heads=2)
#         self.conv3 = GATv2Conv(1024*2, 256, edge_dim=edge_dim, heads=2)

#         self.dropout_layer = torch.nn.Dropout(dropout_prob)

#         self.fc1 = torch.nn.Linear(256*2, 64)
#         self.bn1 = torch.nn.BatchNorm1d(64)
#         self.fc2 = torch.nn.Linear(64, 1)


#     #def forward(self, x, edge_index_prot, edge_attr_prot, edge_index_lig, edge_attr_lig, batch):
#     def forward(self, graphbatch):
        
#         # First convolution only between ligand and protein nodes
#         x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)
#         x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
#         x = F.relu(x)

#         # Pool the nodes of each interaction graph
#         x = global_add_pool(x, batch=graphbatch.batch)

#         # Dropout
#         x = self.dropout_layer(x)

#         # Fully-Connected Layers
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = self.fc2(x)

#         return x