import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, GATv2Conv, global_add_pool, global_max_pool


'''
# MODEL ARCHITECTURE SMALL BASELINE WITH 3 LAYERS AND GLOBAL MEAN POOL 
'''
class GAT0mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT0t(torch.nn.Module): # Not sure about the exact numbers anymore
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0t, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=5)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        self.conv3 = GATv2Conv(512, 1024, edge_dim=edge_dim)

        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class GAT0tmp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

'''BEST ARCHITECTURE SO FAR'''
class GAT0tap(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tap, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
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



'''SMALLER VERSION OF THE BEST ARCHITECTURE'''
class GAT0taps(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0taps, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 128, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(512, 64, edge_dim=edge_dim, heads=4)
        
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
    

'''MEDIUM SIZE VERSION OF THE BEST ARCHITECURE'''
class GAT0tapm(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tapm, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 192, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(768, 64, edge_dim=edge_dim, heads=4)
        
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



'''BEST ARCHITECTURE WITH GLOBAL MEAN AND GLOBAL ADD POOL'''
class GAT0tamp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tamp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 32, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
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
    


'''BEST ARCHITECTURE WITH GLOBAL MEAN AND GLOBAL ADD POOL (original GAT0tap size)'''
class GAT0tampo(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tampo, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
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
    

'''BEST ARCHITECTURE WITH GLOBAL MEAN AND GLOBAL MAX POOL (original GAT0tap size)'''
class GAT0tmmxpo(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tmmxpo, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        xadd = global_max_pool(x, batch=graphbatch.batch)
        xmean = global_mean_pool(x, batch=graphbatch.batch)
        x = torch.cat((xadd,xmean), axis=1)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


'''BEST ARCHITECTURE WITH GLOBAL ADD AND GLOBAL MAX POOL (original GAT0tap size)'''
class GAT0tamxpo(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT0tamxpo, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 64, edge_dim=edge_dim, heads=4)
        
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
        xmean = global_max_pool(x, batch=graphbatch.batch)
        x = torch.cat((xadd,xmean), axis=1)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
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
    
###############################################################################################################################

'''
# MODEL ARCHITECTURE SMALL BASELINE WITH 4 LAYERS AND GLOBAL MEAN POOL 
'''
class GAT04(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT04, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(256, 512, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(512, 1024, edge_dim=edge_dim)
        self.conv4 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv4(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


class GAT04t(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT04t, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 128, edge_dim=edge_dim, heads=5)
        self.conv2 = GATv2Conv(640, 640, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(640, 1024, edge_dim=edge_dim)
        self.conv4 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv4(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



'''
# MODEL ARCHITECTURE SMALL BASELINE WITH 2 LAYERS AND GLOBAL MEAN POOL 
'''
class GAT02(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT02, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 1564, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(1564, 256, edge_dim=edge_dim)

        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

class GAT02t(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT02t, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 782, edge_dim=edge_dim, heads=5)
        self.conv2 = GATv2Conv(3910, 256, edge_dim=edge_dim)

        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



'''
# MODEL ARCHITECTURE SMALL BASELINE WITH 2 LAYERS AND GLOBAL MEAN POOL 
'''
class GAT00mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT00mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 1024, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(1024, 256, edge_dim=edge_dim)

        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    



'''
# MODEL ARCHITECTURE SMALL BASELINE WITH 4 LAYERS AND GLOBAL MEAN POOL 
'''
class GAT000mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT000mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=3)
        self.conv2 = GATv2Conv(1536, 512, edge_dim=edge_dim, heads=2)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim, heads=2)
        self.conv4 = GATv2Conv(512, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
        x = self.conv1(graphbatch.x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv3(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)
        x = self.conv4(x, graphbatch.edge_index, graphbatch.edge_attr)
        x = F.relu(x)

        # Pool the nodes of each interaction graph
        x = global_mean_pool(x, batch=graphbatch.batch)
        x = self.dropout_layer(x)

        # Fully-Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x



'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH AVERAGING) IN THE FIRST LAYER AND GLOBAL MEAN POOL 
'''
class GAT1mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT1mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=2, concat=False)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH AVERAGING) IN THE FIRST TWO LAYERS AND GLOBAL MEAN POOL 
'''
class GAT2mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT2mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=2, concat=False)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim, heads=2, concat=False)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x




'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH AVERAGING) IN ALL THREE LAYERS AND GLOBAL MEAN POOL 
'''
class GAT3mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT3mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 512, edge_dim=edge_dim, heads=2, concat=False)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim, heads=2, concat=False)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim, heads=2, concat=False)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x



'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH CONCAT) IN THE FIRST LAYER AND GLOBAL MEAN POOL 
'''
class GAT1ccmp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT1ccmp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=2)
        self.conv2 = GATv2Conv(512, 1024, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH CONCAT) IN THE FIRST TWO LAYERS AND GLOBAL MEAN POOL 
'''
class GAT2ccmp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT2ccmp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=2)
        self.conv2 = GATv2Conv(512, 512, edge_dim=edge_dim, heads=2)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    




'''
# MODEL ARCHITECTURE SMALL WITH TWO ATTENTION HEADS (WITH CONCAT) IN ALL THREE LAYERS AND GLOBAL MEAN POOL 
'''
class GAT3ccmp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT3ccmp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=2)
        self.conv2 = GATv2Conv(512, 512, edge_dim=edge_dim, heads=2)
        self.conv3 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=2)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    






'''
# MODEL ARCHITECTURE SMALL WITH 3 ATTENTION HEADS (WITH CONCAT) IN THE FIRST LAYER AND GLOBAL MEAN POOL 
'''
class GAT4mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT4mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 512, edge_dim=edge_dim)
        self.conv3 = GATv2Conv(512, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    


'''
# MODEL ARCHITECTURE SMALL WITH SEVERAL ATTENTION HEADS (WITH CONCAT) IN THE FIRST TWO LAYERs AND GLOBAL MEAN POOL 
'''
class GAT5mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT5mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads = 4)
        self.conv3 = GATv2Conv(512, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    




'''
# MODEL ARCHITECTURE SMALL WITH 4 ATTENTION HEADS (WITH CONCAT) IN ALL THREE LAYERS AND GLOBAL MEAN POOL 
'''
class GAT6mp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT6mp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, 256, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(1024, 128, edge_dim=edge_dim, heads=4)
        self.conv3 = GATv2Conv(512, 64, edge_dim=edge_dim, heads=4)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

'''
# MODEL ARCHITECTURE SMALL WITH 4 ATTENTION HEADS (WITH CONCAT) IN ALL THREE LAYERS AND GLOBAL MEAN POOL 
'''
class GAT6lmp(torch.nn.Module):
    def __init__(self, dropout_prob, in_channels, edge_dim):
        super(GAT6lmp, self).__init__()

        #Convolutional Layers
        self.conv1 = GATv2Conv(in_channels, in_channels, edge_dim=edge_dim, heads=4)
        self.conv2 = GATv2Conv(in_channels*4, 256, edge_dim=edge_dim, heads=4)
        self.conv3 = GATv2Conv(1024, 256, edge_dim=edge_dim)
        self.dropout_layer = torch.nn.Dropout(dropout_prob)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, graphbatch):
        
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
        x = F.relu(x)
        x = self.fc2(x)
        return x