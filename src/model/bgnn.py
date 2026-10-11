import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class BGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(BGNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 8)
        self.mlp = Sequential(
            Linear(16, 8),
            ReLU(),
            Linear(8, 2)
        )

    def forward(self, x0, edge_index0, x1, edge_index1, batch0, batch1):
        # 1. Obtain node embeddings
        x0 = self.conv1(x0, edge_index0)
        x0 = x0.relu()
        x0 = self.conv2(x0, edge_index0)
        x0 = x0.relu()
        x0 = self.conv3(x0, edge_index0)
        # 2. Obtain node embeddings
        x1 = self.conv1(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index1)
        x1 = x1.relu()
        x1 = self.conv3(x1, edge_index1)
        # 3. Readout layer
        x0 = global_mean_pool(x0, batch0)  # [batch_size, hidden_channels]
        x1 = global_mean_pool(x1, batch1)  # [batch_size, hidden_channels]
        # 4. Concatenate
        x = torch.cat([x0, x1], dim=1)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.mlp(x)

        return x

def BGNNTrain(model, trainloader, optimizer, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    lossTrain = 0
    for data in trainloader:
        optimizer.zero_grad()
        data.to(device)
        out = model.forward(data[a_nodes], data[a_edges], data[b_nodes], data[b_edges], data[a_batch], data[b_batch])
        loss = criterion(out, data[label])
        loss.backward() 
        optimizer.step() 
        lossTrain += loss.item() * len(data[label])

    lossTrain /= len(trainloader.dataset)

    return model, lossTrain