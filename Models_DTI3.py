import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, GCNConv
from torch_geometric.data import Batch

    

'''BEST ARCHITECTURE SO FAR'''
class GAT0tap(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tap, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=0.1)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=0.1)
        
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
    
    
'''BEST ARCHITECTURE WITH GLOBAL MEAN AND GLOBAL ADD POOL (original GAT0tap size)'''
class GAT0tampo(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tampo, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4, dropout=0.1)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4, dropout=0.1)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        xadd = global_add_pool(x, batch=graphbatch.batch)
        xmean = global_mean_pool(x, batch=graphbatch.batch)
        x = torch.cat((xadd,xmean), axis=1)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


'''BEST ARCHITECTURE TRY POOLING LAYER WITH MASTER NODE'''

class GAT0tmaster(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmaster, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class GAT0tmasterb(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmasterb, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 64, edge_dim=edge_dim, heads=6)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(384, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT0tmasterc(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmasterc, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 64, edge_dim=edge_dim, heads=8)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



class GAT0tmaster2(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmaster2, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 128, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT0tmaster2b(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmaster2b, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 128, edge_dim=edge_dim, heads=6)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(768, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT0tmaster2c(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmaster2c, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 128, edge_dim=edge_dim, heads=8)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT0tmaster3(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmaster3, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.poolconv = GATv2Conv(256, 256, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(1024, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.poolconv(x, graphbatch.edge_index_master)

        # Pool the nodes of each interaction graph
        
        last_node_indeces = graphbatch.n_nodes.cumsum(dim=0) - 1
        master_node_features = x[last_node_indeces]

        x = self.dropout_layer(master_node_features)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x










# --------------------------------------------------------------------------------------------------------------------------------

'''BEST ARCHITECTURE SO FAR WITH ONLY ONE FC LAYER'''
class GAT0tapfc1(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tapfc1, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc = torch.nn.Linear(256, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc(x)
        x = F.relu(x)

        return x
    
    
class WishfulThinking(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(WishfulThinking, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(in_channels, 128, edge_dim=edge_dim, heads=4)
        self.conv3 = GATv2Conv(512, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        
        x = global_add_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class WishfulThinking_JK(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(WishfulThinking_JK, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(in_channels, 128, edge_dim=edge_dim, heads=4)
        self.conv3 = GATv2Conv(512, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512+256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index_lig, graphbatch.edge_attr_lig)
        x = F.relu(x)

        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x1 = global_add_pool(x, batch=graphbatch.batch)

        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x2 = global_add_pool(x, batch=graphbatch.batch)

        x = torch.cat((x1,x2), axis=1)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
