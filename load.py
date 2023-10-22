from config import device, script_dir
import torch
import dgl
import csv
import os
import numpy as np

def transform_graph(edge_info, node_info):
    g = dgl.graph((edge_info[:, 0].to(torch.int64),edge_info[:, 1].to(torch.int64)))
    features = node_info[:, 1:]
    g.ndata['features'] = features.to(torch.float32)
    g.ndata['features'] = torch.cat((g.ndata['features'], torch.rand(g.number_of_nodes(), 10)), dim=1)
    g.edata['distance'] = edge_info[:, 2].to(torch.float32)
    return g.to(device)

def find_vote(graph):
    to_vote = []
    for node_id, features in enumerate(graph.ndata['features']):
        if features[0] == -1:
            to_vote.append(node_id)
    return to_vote

def read_csv_file(file_path):
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(row)
    return data

def save_vote_result_to_csv(episode, vote_node, vote_result):
    with open(script_dir + '/vote_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [episode, vote_node, vote_result.item()]
        writer.writerow(row)

def save_score(method, density, diversity, proximity, global_reward, prosocial, eposode = 0, only_top_down = False):
    density = np.array(density.cpu())
    diversity = np.array(diversity.cpu())
    proximity = np.array(proximity.cpu())
    global_reward = global_reward.item()
    prosocial = prosocial.item()

    if method == 1:
        if only_top_down:
            file_path = script_dir + '/greedy_score_only_top.csv'
        else:
            file_path = script_dir + '/greedy_score.csv'
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = ['density_Residential', 'density_Office', 'density_Commercial', 'density_Facility', 'density_Greenspace', 'diversity', 'proximity_Residential', 'proximity_Office', 'proximity_Commercial', 'proximity_Facility', 'proximity_Greenspace', 'global_reward', 'prosocial']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            row = {
                'density_Residential': density[0],
                'density_Office': density[1],
                'density_Commercial': density[2],
                'density_Facility': density[3],
                'density_Greenspace': density[4],
                'diversity': diversity,
                'proximity_Residential': proximity[0],
                'proximity_Office': proximity[1],
                'proximity_Commercial': proximity[2],
                'proximity_Facility': proximity[3],
                'proximity_Greenspace': proximity[4],
                'global_reward': global_reward,
                'prosocial': prosocial,
            }
            writer.writerow(row)

    if method == 2:
        file_path = script_dir + '/random_score.csv'
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = ['density_Residential', 'density_Office', 'density_Commercial', 'density_Facility', 'density_Greenspace', 'diversity', 'proximity_Residential', 'proximity_Office', 'proximity_Commercial', 'proximity_Facility', 'proximity_Greenspace', 'global_reward', 'prosocial']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            row = {
                'density_Residential': density[0],
                'density_Office': density[1],
                'density_Commercial': density[2],
                'density_Facility': density[3],
                'density_Greenspace': density[4],
                'diversity': diversity,
                'proximity_Residential': proximity[0],
                'proximity_Office': proximity[1],
                'proximity_Commercial': proximity[2],
                'proximity_Facility': proximity[3],
                'proximity_Greenspace': proximity[4],
                'global_reward': global_reward,
                'prosocial': prosocial,
            }
            writer.writerow(row)
    
    if method == 3:
        if only_top_down:
            file_path = script_dir + '/MARL_score_only_top.csv'
        else:
            file_path = script_dir + '/MARL_score.csv'
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as csvfile:
            fieldnames = ['eposode', 'density_Residential', 'density_Office', 'density_Commercial', 'density_Facility', 'density_Greenspace', 'diversity', 'proximity_Residential', 'proximity_Office', 'proximity_Commercial', 'proximity_Facility', 'proximity_Greenspace', 'global_reward', 'prosocial']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            row = {
                'eposode': eposode,
                'density_Residential': density[0],
                'density_Office': density[1],
                'density_Commercial': density[2],
                'density_Facility': density[3],
                'density_Greenspace': density[4],
                'diversity': diversity,
                'proximity_Residential': proximity[0],
                'proximity_Office': proximity[1],
                'proximity_Commercial': proximity[2],
                'proximity_Facility': proximity[3],
                'proximity_Greenspace': proximity[4],
                'global_reward': global_reward,
                'prosocial': prosocial,
            }
            writer.writerow(row)