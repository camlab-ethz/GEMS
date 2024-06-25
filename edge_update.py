import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader



class EdgeFeatureUpdateNetwork(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, out_dim):
        super(EdgeFeatureUpdateNetwork, self).__init__()
        
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



# Example usage with PyTorch Geometric's Data object

dataset = torch.load('dataset_casf2013.pt')
train_loader = DataLoader(dataset = dataset, batch_size=16)

data = next(iter(train_loader))

print("Initialize EdgeUpdateNetwork")
edge_update_net = EdgeFeatureUpdateNetwork(node_feature_dim=60, edge_feature_dim=7, hidden_dim=64, out_dim=16)

print("Update Edges")
updated_edge_attr = edge_update_net(data.edge_index, data.x, data.edge_attr)





# class GCNWithEdgeUpdates(nn.Module):
#     def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim, out_dim, num_classes):
#         super(GCNWithEdgeUpdates, self).__init__()
        
#         self.edge_update_net = EdgeFeatureUpdateNetwork(node_feature_dim, edge_feature_dim, hidden_dim, hidden_dim)
#         self.gcn1 = GCNConv(node_feature_dim, hidden_dim)
#         self.gcn2 = GCNConv(hidden_dim, out_dim)
#         self.classifier = nn.Linear(out_dim, num_classes)

#     def forward(self, data):
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         # Update edge features
#         updated_edge_attr = self.edge_update_net(edge_index, x, edge_attr)

#         # GCN Layer 1
#         x = self.gcn1(x, edge_index)
#         x = F.relu(x)

#         # Optionally, update edge features again or use the previous ones
#         updated_edge_attr = self.edge_update_net(edge_index, x, updated_edge_attr)

#         # GCN Layer 2
#         x = self.gcn2(x, edge_index)

#         # Classifier
#         x = self.classifier(x)

#         return F.log_softmax(x, dim=1)




# # Example usage with PyTorch Geometric's Data object
# from torch_geometric.data import Data

# num_nodes = 10
# num_edges = 45
# node_feature_dim = 32
# edge_feature_dim = 60
# hidden_dim = 128
# out_dim = 128
# num_classes = 3

# node_features = torch.randn(num_nodes, node_feature_dim)
# edge_features = torch.randn(num_edges, edge_feature_dim)
# edge_index = torch.randint(0, num_nodes, (2, num_edges))
# labels = torch.randint(0, num_classes, (num_nodes,))

# data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_features, y=labels)

# model = GCNWithEdgeUpdates(node_feature_dim, edge_feature_dim, hidden_dim, out_dim, num_classes)
# output = model(data)

# print(output.shape)  # Should be (num_nodes, num_classes)