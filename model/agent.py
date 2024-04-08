import torch
import torch.nn as nn
import dgl
import random
import dgl.nn as dglnn
import torch.nn.functional as F

device = torch.device("cuda:0")

# Actor Network
class Actor(nn.Module):
    
    def __init__(self, area_graph, in_dim, hidden_dim, out_dim, num_heads=1):
        super(Actor, self).__init__()
        self.layer1 = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.layer2 = dglnn.GATConv(hidden_dim * num_heads, out_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.single_graph = area_graph

    def forward(self, x, batch = False, exploration_momentum = 0.5, explore = False):
        # Input a batch during training
        if batch:
            batch_size = x.shape[0]
            graphs = [self.single_graph for _ in range(batch_size)]
            batched_graph = dgl.batch(graphs)       
            x = self.layer1(batched_graph, x.view(-1, x.size(-1)))
            x = self.layer2(batched_graph, x)

        else:
            x = self.layer1(self.single_graph, x)
            x = self.layer2(self.single_graph, x)

        x = x.squeeze()
        x =  torch.softmax(x, dim=-1)
        x = x.squeeze()
        
        # Motivate agent to explore more action
        if explore:
            if random.random() < exploration_momentum:
                x = torch.rand_like(x)
        
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=1):
        super(Critic, self).__init__()
        self.layer1 = dglnn.GATConv(in_dim, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.layer2 = dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, observation_graphs, batch=False):
        if batch:
            batched_graph = dgl.batch(observation_graphs)
            h = batched_graph.ndata['features']
            h = F.relu(self.layer1(batched_graph, h))
            h = F.relu(self.layer2(batched_graph, h))
            batched_graph.ndata['h'] = h

            hg = dgl.mean_nodes(batched_graph, 'h')

            hg = hg.squeeze()

            return self.fc(hg)
        else:
            h = observation_graphs.ndata['features']
            h = F.relu(self.layer1(observation_graphs, h))
            h = F.relu(self.layer2(observation_graphs, h))

            x = x.squeeze()

            return self.fc(h.mean(0, keepdim=True))
    
