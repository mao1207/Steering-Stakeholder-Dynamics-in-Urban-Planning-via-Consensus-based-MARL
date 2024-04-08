from multi_agent_env import MultiAgentEnvironment
from gym_env import Land
import load  
import torch
from greedy import greedy
import config 
import argparse
import random
import dgl
import csv
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy
import argparse

device = torch.device("cuda:0")

# def generate_data(num_nodes=10, num_edges=15):
#     node_info = []
#     edge_info = []
    
#     for i in range(num_nodes):
#         use = random.randint(-1, 4)
#         space = random.randint(1, 10)
#         node_info.append([use, space])
        
#     for i in range(num_edges):
#         src = random.randint(0, num_nodes-1)
#         dst = random.randint(0, num_nodes-1)
#         while dst == src:
#             dst = random.randint(0, num_nodes-1)
#         distance = random.randint(10, 200)
#         edge_info.append([src, dst, distance])

#     return node_info, edge_info


node_info_path = os.path.join(config.script_dir, 'node_info.csv')
edge_info_path = os.path.join(config.script_dir, 'node_pairs_knn4.csv')
# cluster_info_path = os.path.join(config.script_dir, 'node_info_cluster13.csv')

node_info_data = np.array(load.read_csv_file(node_info_path), dtype=float)
edge_info_data = np.array(load.read_csv_file(edge_info_path), dtype=float)
# cluster_info_data = np.array(load.read_csv_file(cluster_info_path), dtype=int)
# cluster_info_data = load.cluster_nodes(cluster_info_data)



def main():
    node_info, edge_info = node_info_data, edge_info_data
    node_tensor = torch.tensor(node_info)
    edge_tensor = torch.tensor(edge_info)
    land = Land(node_tensor, edge_tensor)
    area_graph = load.transform_graph(edge_tensor, node_tensor)

    env = MultiAgentEnvironment(area_graph, config.args.n_inp, config.args.n_hid, 5, load.find_vote(area_graph))

    if config.args.experiment_mode == "actor-critic":
        with open(config.script_dir + '/vote_results.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Episode'] + ['node ID'] + ['Result']
            writer.writerow(header)

        state_begin = area_graph.ndata['features']

        to_vote = load.find_vote(area_graph)

        for episode in range(config.args.n_epoch):
            explore = False
            if episode < 40:
                explore = True

            state = copy.deepcopy(state_begin)

            # nx_g = area_graph.cpu().to_networkx()
            # pos = nx.spring_layout(nx_g)
            # color_list = ['gray', 'red', 'blue', 'green', 'orange', 'purple']
            # pos = nx.spring_layout(nx_g)
            # color_indices = state[:, 0].tolist()
            # node_colors = [color_list[idx + 1] for idx in color_indices]
            # nx.draw(nx_g, pos, with_labels=False, node_color=node_colors, node_size=5, width=0.5, edge_color='lightgray', arrows=False)
            # plt.savefig(script_dir + 'graph_visualization.png')
            
            for vote_node in to_vote:

                vote_results = {}
                agent_class_per_vote = []
                state_per_vote = []
                action_per_vote = []
                reward_per_vote = []
                next_state_per_vote = []
                observation_per_vote = []
                arrive_nodes_per_vote = []
                agent_vote = []

                for agent_class in range(5):
                    class_votes = []
                    for agent_idx in range(env.agents_per_class[agent_class]):
                        with torch.no_grad():
                            arrive_nodes = land.get_observation(env.locates[agent_class][agent_idx], config.max_distance).to(config.device)
                            agent_locate = torch.empty(area_graph.number_of_nodes(), 1).fill_(0).to(config.device)
                            agent_locate[arrive_nodes] = 1
                            agent_locate[env.locates[agent_class][agent_idx]] = 2
                            
                            
                            state = torch.cat((state, config.locate_one_hot[agent_locate.long().view(-1)]), dim=1)
                            q_values = env.agents[agent_class](state, explore=explore, exploration_momentum = 0.5-episode*0.0125)
                            action = torch.argmax(q_values[vote_node], dim=0).int().item()
                            class_votes.append(action)
                            arrive_nodes_per_vote.append(arrive_nodes)
                            if_vote = torch.empty(area_graph.number_of_nodes(), 1).fill_(-1).to(config.device)
                            if_vote[to_vote] = 0
                            if_vote[vote_node] = 1
                            area_graph.ndata['features'] = torch.cat((area_graph.ndata['features'], q_values), dim=1)
                            area_graph.ndata['features'] = torch.cat((area_graph.ndata['features'], if_vote), dim=1)
                            if agent_class < 2:
                                observation = copy.deepcopy(area_graph)
                            else:
                                observation = area_graph.subgraph(arrive_nodes)
                            area_graph.ndata['features'] = area_graph.ndata['features'][:, :14]

                        agent_class_per_vote.append(agent_class)
                        state_per_vote.append(state)
                        action_per_vote.append(q_values)
                        observation_per_vote.append(observation)
                        agent_vote.append(action)
                        state = state[:, :14]

                    vote_results[agent_class] = class_votes
                    # print("vote_results",vote_results)
                    # print("vote_results[agent_class]",vote_results[agent_class])

                vote_result = env.vote(vote_results, land = land, only_top_down = config.args.if_only_top_down)
                print("vote_results",vote_results)
                state[vote_node, 0] = vote_result
                area_graph.ndata['features'] = state
                # print('==========================')
                # # print(agent_vote)
                # print("vote_result")
                
                # print(vote_result)
                # print('==========================')

                if config.args.if_only_top_down:
                    agent_vote = [0 for x in agent_vote]
                else:
                    agent_vote = [land.focus_indices_origin[agent_class][vote_result].item() if x == vote_result else 0 for x in agent_vote]

                vote_success_per_class = [0] * 5
                vote_success_per_class[0] = agent_vote[0] * 15
                vote_success_per_class[1] = agent_vote[1] * 15
                vote_success_per_class[2] = sum(agent_vote[2:12])
                vote_success_per_class[3] = sum(agent_vote[12:22])
                vote_success_per_class[4] = sum(agent_vote[22:32])

                a = [0] * 5
                
                for i in range(len(agent_class_per_vote)):
                    next_state_per_vote.append(area_graph.ndata['features'])
                    observation_after_vote = area_graph.subgraph(arrive_nodes_per_vote[i])
                    score = land.calculate_reward(agent_class_per_vote[i], observation_per_vote[i], observation_after_vote, agent_vote[i], area_graph, vote_success_per_class, vote_result.item())
                    reward_per_vote.append(score)
                    a[agent_class_per_vote[i]] += score.item()

                # print(a)

                env.push_to_buffer(agent_class_per_vote, state_per_vote, action_per_vote, reward_per_vote, next_state_per_vote, observation_per_vote)

                load.save_vote_result_to_csv(episode, vote_node, vote_result)

                for agent_class in range(5):
                    env.update_network(agent_class, batch_size=64)

                # get urban performance score
                # for i in range(13):
                #     density, diversity, proximity = land.get_urban_performance_score(area_graph, cluster_info_data[i])
                #     print(i, density, diversity, proximity)

                print("Epoch:%d, vote_node_id:%d, vote_result:%d"%(episode, vote_node, vote_result))

            print(f"Episode {episode + 1}")
            
            agent_score = [0] * 5
            
            for agent_class in range(5):
                agent_score[agent_class] = 0
                for agent_idx in range(env.agents_per_class[agent_class]):
                    arrive_nodes = land.get_observation(env.locates[agent_class][agent_idx], config.max_distance).to(config.device)
                    if agent_class < 2:
                        observation = copy.deepcopy(area_graph)
                    else:
                        observation = area_graph.subgraph(arrive_nodes)
                    agent_score[agent_class] += land.get_agent_score(agent_class, observation).item()
                agent_score[agent_class] /= env.agents_per_class[agent_class]

            density, diversity, proximity = land.get_urban_performance_score(area_graph)
            load.save_score(3, density, diversity, proximity, land.get_global_score(area_graph), land.get_prosocial_score(agent_score), episode, only_top_down = config.args.if_only_top_down)
            
            print(agent_score)
            print(land.get_global_score(area_graph))
            print(land.get_prosocial_score(agent_score))
                    
            
    
    elif config.args.experiment_mode == "greedy":
        vote_result = greedy(env,land ,area_graph, only_top_down = config.args.if_only_top_down)

        agent_score = [0] * 5
            
        for agent_class in range(5):
            agent_score[agent_class] = 0
            for agent_idx in range(env.agents_per_class[agent_class]):
                arrive_nodes = land.get_observation(env.locates[agent_class][agent_idx], config.max_distance).to(config.device)
                if agent_class < 2:
                    observation = copy.deepcopy(area_graph)
                else:
                    observation = area_graph.subgraph(arrive_nodes)
                agent_score[agent_class] += land.get_agent_score(agent_class, observation).item()
            agent_score[agent_class] /= env.agents_per_class[agent_class]

        density, diversity, proximity = land.get_urban_performance_score(area_graph)
        load.save_score(1, density, diversity, proximity, land.get_global_score(area_graph), land.get_prosocial_score(agent_score), only_top_down = config.args.if_only_top_down)
        
        print(agent_score)
        print(land.get_global_score(area_graph))
        print(land.get_prosocial_score(agent_score))

    elif config.args.experiment_mode == "random":
        density, diversity, proximity, global_score, prosocial_score = torch.empty(0, 5).to(device), torch.empty(0).to(device), torch.empty(0, 5).to(device), torch.empty(0).to(device), torch.empty(0).to(device)

        to_vote = load.find_vote(area_graph)
        for _ in range(20):
            area_graph.ndata['features'][to_vote, 0] = torch.randint(0, 5, (len(to_vote),)).float().to(device)
            print(area_graph.ndata['features'][to_vote, 0])

            agent_score = [0] * 5
                
            for agent_class in range(5):
                agent_score[agent_class] = 0
                for agent_idx in range(env.agents_per_class[agent_class]):
                    arrive_nodes = land.get_observation(env.locates[agent_class][agent_idx], config.max_distance).to(config.device)
                    if agent_class < 2:
                        observation = copy.deepcopy(area_graph)
                    else:
                        observation = area_graph.subgraph(arrive_nodes)
                    agent_score[agent_class] += land.get_agent_score(agent_class, observation).item()
                agent_score[agent_class] /= env.agents_per_class[agent_class]

            density_per_epoch, diversity_per_epoch, proximity_per_epoch = land.get_urban_performance_score(area_graph)
            density = torch.cat((density, density_per_epoch.unsqueeze(0)), dim=0)
            diversity = torch.cat((diversity, diversity_per_epoch.unsqueeze(0)), dim=0)
            proximity = torch.cat((proximity, proximity_per_epoch.unsqueeze(0)), dim=0)

            print('diversity', torch.mean(diversity,dim=0))
            
            global_score = torch.cat((global_score, torch.tensor(land.get_global_score(area_graph)).unsqueeze(0).to(device)), dim=0)
            prosocial_score = torch.cat((prosocial_score, torch.tensor(land.get_prosocial_score(agent_score)).unsqueeze(0).to(device)), dim=0)
            print(density)
            print(global_score)
            print(prosocial_score)
            
            print(agent_score)
            print(land.get_global_score(area_graph))
            print(land.get_prosocial_score(agent_score))

        load.save_score(2, torch.mean(density,dim=0), torch.mean(diversity,dim=0), torch.mean(proximity,dim=0), torch.mean(global_score,dim=0), torch.mean(prosocial_score,dim=0), only_top_down = config.args.if_only_top_down)
        

if __name__ == "__main__":
    main()
