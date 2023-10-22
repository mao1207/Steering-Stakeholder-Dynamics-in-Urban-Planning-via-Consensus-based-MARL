import load
from config import *
import torch
import copy

# def greedy(env, land, area_graph):
#     to_vote = load.find_vote(area_graph)
#     state = area_graph.ndata['features']
#     for vote_node in to_vote:
#         vote_result = {}
#         for agent_class in range(5):
#             class_votes = []
#             class_votes = torch.tensor(class_votes, dtype=torch.long)
#             if agent_class not in vote_result:
#                 vote_result[agent_class] = torch.tensor([], dtype=torch.long)
#             around_focus_indices = land.focus_indices[agent_class]
#             first_iteration = True
#             for agent_idx in range(env.agents_per_class[agent_class]):
#                 if first_iteration:
#                     pre_agent_idx = agent_idx
#                 break
#             first_agent_idx = pre_agent_idx
#             for agent_idx in range(env.agents_per_class[agent_class]):
#                 with torch.no_grad():
#                     around_focus_indices = land.focus_indices[agent_class]
#                     arrive_nodes = land.get_observation(env.locates[agent_class][agent_idx], config.max_distance).to(config.device)
#                     agent_locate = torch.empty(area_graph.number_of_nodes(), 1).fill_(0).to(config.device)
#                     state = area_graph.ndata['features']
#                     land_use = state[:,0][arrive_nodes].int()     
#                     around_focus_indices = personal_focus_indices(first_agent_idx,land,around_focus_indices,arrive_nodes,agent_class,state) 
#                     # land.focus_indices[agent_class] =  around_focus_indices
#                     max_index = vote_one_node(land_use, class_votes,agent_class,around_focus_indices)
#                     class_votes = torch.cat((class_votes, torch.tensor([max_index], dtype=torch.long)))                   
#                 pre_agent_idx = agent_idx               
#             class_votes = [int(x.item()) for x in class_votes]        
#             class_votes = torch.tensor(class_votes, dtype=torch.long)       
#             vote_result[agent_class] = torch.cat((vote_result[agent_class], class_votes), dim=0)        
#         vote_result_num = env.vote(vote_result)        
#         state[vote_node, 0] = vote_result_num
#         area_graph.ndata['features'] = state
#     return vote_result_num

# def personal_focus_indices(first_agent_idx,land,around_focus_indices,arrive_nodes,agent_class,state):
#     random_tensor =torch.rand(around_focus_indices.shape).to(config.device)
#     around_focus_indices = around_focus_indices + random_tensor
#     land_use = state[:,0][arrive_nodes].int()  
#     land_use = torch.where(land_use < 0, 5, land_use)
#     land_use_type_count =  torch.bincount(land_use, minlength = 5)[:5]
#     diff_land_use_type_count = land_use_type_count-first_agent_idx
#     adjustments = torch.pow(0.99, diff_land_use_type_count.float()).to(config.device)
#     around_focus_indices = around_focus_indices * adjustments
#     return around_focus_indices

# def vote_one_node(land_use,class_votes,agent_class,around_focus_indices):
#     max_value, max_index = torch.max(around_focus_indices, 0)
#     class_votes = torch.cat((class_votes, torch.tensor([max_index], dtype=torch.long)))
#     return max_index

def greedy(env, land, area_graph, only_top_down = False):
    to_vote = load.find_vote(area_graph)
    state_begin = area_graph.ndata['features']
    state = copy.deepcopy(state_begin)

    for vote_node in to_vote:
        vote_results = {}

        for agent_class in range(5):
            class_votes = []

            for agent_idx in range(env.agents_per_class[agent_class]):

                observe = land.get_observation(env.locates[agent_class][agent_idx], max_distance)
                land_use_type = state[observe, 0].int()
                land_use_type = torch.where(land_use_type < 0, 5, land_use_type)
                land_use_count = torch.bincount(land_use_type, minlength=5)[:5]
                land_use_type_begin = state_begin[observe, 0].int()
                land_use_type_begin = torch.where(land_use_type_begin < 0, 5, land_use_type_begin)
                land_use_count_begin = torch.bincount(land_use_type_begin, minlength=5)[:5]
                land_use_count_add = land_use_count - land_use_count_begin

                vote_preference = land.focus_indices[agent_class] * (0.8 ** land_use_count_add)
                vote_preference += torch.randn_like(vote_preference) * 0.3
                class_votes.append(torch.argmax(vote_preference).item())

            vote_results[agent_class] = class_votes

        vote_result = env.vote(vote_results, only_top_down)
        state[vote_node, 0] = vote_result
        print("vote_node_id:%d, vote_result:%d"%(vote_node, vote_result))

    area_graph.ndata['features'][:, 0] = state[:, 0]


