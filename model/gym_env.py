import torch
from collections import deque

import torch
from collections import deque
import numpy as np
from math import *
import heapq

device = torch.device("cuda:0")

class Node():
    def __init__(self, id):
        self.id = id
        self.use = 0
        self.neighbors = {}

    def add_neighbor(self, neighbor, distance=1):
        self.neighbors[neighbor] = distance


class Land():
    
    def __init__(self, node_info, edge_info):
        self.nodes = [None] * node_info.shape[0]
        for i in range(node_info.shape[0]):
            self.nodes[i] = Node(i)
            self.nodes[i].use = node_info[i][1].item()
            self.nodes[i].space = node_info[i][2].item()

        for i in range(edge_info.shape[0]):
            self.nodes[edge_info[i][0].int().item()].add_neighbor(edge_info[i][1].int().item(), edge_info[i][2].item())
            self.nodes[edge_info[i][1].int().item()].add_neighbor(edge_info[i][0].int().item(), edge_info[i][2].item())

        # notice: the order of landuse type is Residential, Office, Commercial, Facility, Greenspace
        self.focus_indices= torch.tensor([
            [1, 1, 1, 0.5, 0.5], # developer
            [1, 1, 1, 1, 1], # urban planner
            [0, 1, 1, 0, 0.5], # high income
            [1, 0.5, 1, 0.5, 0.5], # middle income
            [1, 0, 0.5, 1, 0.5], # low income
        ]).to(device)

        self.focus_indices_origin= torch.tensor([
            [1, 1, 1, 0.5, 0.5], # developer
            [1, 1, 1, 1, 1], # government
            [0, 1, 1, 0, 0.5], # high income
            [1, 0.5, 1, 0.5, 0.5], # middle income
            [1, 0, 0.5, 1, 0.5], # low income
        ]).to(device)

        
    def count_landuses(self, observation):
        landuse_type = observation.ndata['features'][:, 0]
        landuse_type = torch.where(landuse_type == -1, torch.tensor(5), landuse_type)
        counter = torch.bincount(landuse_type.long(), minlength=5)

        return counter[0:5].float()
    
    def local_awareness(self, agent_class, observation_before_vote, observation_after_vote):
        weight = torch.tensor(self.focus_indices[agent_class]).float()
        # print(weight)
        # print(self.count_landuses(observation_after_vote) - self.count_landuses(observation_before_vote))
        # print(weight.dot(self.count_landuses(observation_after_vote) - self.count_landuses(observation_before_vote)))
        # weight = torch.tensor(self.focus_indices[agent_class]).float() - (2/3) * torch.log(self.satisfied[agent_class] + 1).float()
        return weight.dot(self.count_landuses(observation_after_vote) - self.count_landuses(observation_before_vote)) # this could be optimized by cutting a round of traversal
    
    def global_awareness(self, area_graph):
        landuses_count = torch.zeros(5)
        facility_area = 0
        commercial_area = 0
        greenspace_area = 0
        landuse_type = area_graph.ndata['features'][:, 0]
        landuse_type = torch.where(landuse_type == -1, torch.tensor(5), landuse_type)
        area = area_graph.ndata['features'][:, 1]
        total_area = torch.sum(area)
        landuses_count = torch.bincount(landuse_type.long(), minlength=5)
        facility_area = torch.sum(area[landuse_type == 3])
        commercial_area = torch.sum(area[landuse_type == 2])
        greenspace_area = torch.sum(area[landuse_type == 4])

        landuses_percentage = landuses_count / landuses_count.sum()
        landuses_percentage =torch.where(landuses_percentage == 0, 1, landuses_percentage)
        diversity = - (landuses_percentage * torch.log(landuses_percentage)).sum()
        
        density = 0.0
        if facility_area != 0:
            density += (landuses_count[3]) / total_area
        if commercial_area != 0:
            density += (landuses_count[2]) / total_area
        if greenspace_area != 0:
            density += (landuses_count[4]) / total_area

        return torch.tensor(density*1000 + diversity)
    
    def social_awareness(self, success_vote):
        return - np.var(success_vote[2:]) - fabs(success_vote[0] - success_vote[1])
    
    def calculate_reward(self, agent_class, observation_before_vote, observation_after_vote, self_awareness, global_graph, vote_success_per_class, vote_result):
        # print("=====================")
        # print(self.local_awareness(agent_class, observation_before_vote, observation_after_vote))
        # print(self_awareness)
        # print(self.social_awareness(vote_success_per_class)/10)
        # print(self.global_awareness(global_graph))
        if agent_class < 2:
            return self.local_awareness(agent_class, observation_before_vote, observation_after_vote) + self_awareness + self.global_awareness(global_graph)
        else:
            return self.local_awareness(agent_class, observation_before_vote, observation_after_vote) + self_awareness + self.global_awareness(global_graph) + self.social_awareness(vote_success_per_class)/20


    def get_observation(self, id, max_distance):
        start_node = self.nodes[id]
        visited = torch.empty(0, dtype=torch.long)
        queue = deque([(start_node, 0)])
        
        while queue:
            current_node, accumulated_distance = queue.popleft()
            
            if accumulated_distance > max_distance:
                continue
                
            if current_node.id not in visited.tolist():
                visited = torch.cat((visited, torch.tensor([current_node.id], dtype=torch.long)), dim = 0)
            else:   
                continue;

            for neighbor, distance_to_neighbor in current_node.neighbors.items():
                if neighbor not in visited.tolist():
                    new_accumulated_distance = accumulated_distance + distance_to_neighbor
                    queue.append((self.nodes[neighbor], new_accumulated_distance))
        
        return visited

    def get_urban_performance_score(self, area_graph): # node_ids is a list of node ids
        # there are three metrics: density, diversity, proximity
        # these metrics are calculated among the nodes in node_ids, which is a cluster of parcles

        # 1. calculate diversity
        # landuse_type = area_graph.ndata['features'][:, 0]
        # landuse_type = torch.where(landuse_type == -1, torch.tensor(5), landuse_type)
        # landuses_count = torch.bincount(landuse_type.long(), minlength=5)
        # landuses_count = landuses_count[:5] # get rid of the -1
        # landuses_percentage = landuses_count / landuses_count.sum()
        # landuses_percentage =torch.where(landuses_percentage == 0, 1, landuses_percentage)
        # diversity = - (landuses_percentage * torch.log(landuses_percentage)).sum()
        landuse_type = area_graph.ndata['features'][:, 0]
        landuse_type = torch.where(landuse_type == -1, torch.tensor(5), landuse_type)
        landuses_count = torch.bincount(landuse_type.long(), minlength=5)
        landuses_count = landuses_count[:5] # get rid of the -1s
        landuses_percentage = landuses_count / landuses_count.sum()
        landuses_percentage =torch.where(landuses_percentage == 0, 1, landuses_percentage)
        diversity = - (landuses_percentage * torch.log(landuses_percentage)).sum()

        # 2. calculate density
        # area = area_graph.ndata['features'][:, 1]
        # sum_of_each_area = torch.Tensor([ torch.sum(area[landuse_type == i]) for i in range(5) ]) # might have a torch style to do this
        # landuse_type = area_graph.ndata['features'][:, 0]
        # landuse_type = torch.where(landuse_type == -1, torch.tensor(5), landuse_type)
        # sum_of_each_area = torch.where(sum_of_each_area == 0, 1, sum_of_each_area).to(device)
        # density = landuses_count /  sum_of_each_area
        area = area_graph.ndata['features'][:, 1]
        sum_of_each_area = torch.Tensor([ torch.sum(area[landuse_type == i]) for i in range(5) ]) # might have a torch style to do this
        sum_of_each_area = torch.where(sum_of_each_area == 0, 1, sum_of_each_area).to(device)
        density = landuses_count /  sum_of_each_area

        # 3. calculate proximity
        # need to traverse every node
        # use dijkstra algorithm to find the shortest path
        proximity = torch.zeros(5)

        for idx in range(landuse_type.shape[0]):
            node = self.nodes[int(idx)]
            distances = {node: float('inf') for node in self.nodes}
            distances[node] = 0

            # Priority queue for selecting the next node to explore
            priority_queue = [(0, node)]

            landuse_type_found = [False] * 5
            
            while priority_queue:

                current_distance, current_node = heapq.heappop(priority_queue)
                current_landuse_type = int(area_graph.ndata['features'][current_node.id, 0])
                if current_landuse_type != -1: 
                    landuse_type_found[current_landuse_type] = True

                # if all landuse types are found, then break
                if landuse_type_found == [True] * 5: # no cuda style
                    break
                # if torch.Tensor(landuse_type_found).all(): # cuda style
                #     break

                # Check if this distance is outdated
                # print("current_node",current_node)
                if current_distance > distances[current_node]:
                    continue

                for neighbor, distance_to_neighbor in current_node.neighbors.items():
                    neighbor = self.nodes[neighbor]
                    distance = distances[current_node] + distance_to_neighbor

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue, (distance, neighbor))
            
            dist = torch.Tensor([1250] * 5) 
            for node, distance in distances.items():
                current_landuse_type = int(area_graph.ndata['features'][node.id, 0])
                if current_landuse_type != -1 and distance < dist[ int(current_landuse_type) ]:
                    dist[ int(current_landuse_type) ] = distance
                
            proximity = proximity + dist
        
        proximity = torch.where(proximity == 0, 1, proximity).to(device)
        proximity = 1 / proximity

        return (density, diversity, proximity)
    
    def get_agent_score(self, agent_class, observation):
        weight = torch.tensor(self.focus_indices[agent_class]).float()
        agent_benefit = weight.dot(self.count_landuses(observation)) 
        return agent_benefit

    def get_global_score(self, area_graph):
        return self.global_awareness(area_graph)
    
    def get_prosocial_score(self, agent_benefit):
        return np.var(agent_benefit[2:]) + fabs(agent_benefit[0] - agent_benefit[1])
    


        
