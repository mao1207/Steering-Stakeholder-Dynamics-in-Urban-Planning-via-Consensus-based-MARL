import load
from config import *
import torch
import copy

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


