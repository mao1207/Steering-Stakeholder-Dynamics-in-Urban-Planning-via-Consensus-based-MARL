import torch
import torch.optim as optim
from agent import Actor, Critic
import random
import copy

device = torch.device("cuda:0")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, observation):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, observation)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class MultiAgentEnvironment:
    def __init__(self, area_graph, in_dim, hidden_dim, out_dim, to_vote):
        self.agent_classes = 5
        self.agents_per_class = [1, 1, 10, 10, 10]  
        self.vote_weight = [15, 15, 1, 1, 1]
        self.agents = []
        self.critics = []
        self.buffers = []
        self.optimizers_actor = []
        self.optimizers_critic = []
        self.locates = torch.randint(0, area_graph.number_of_nodes(), (5, 10))
        self.area_graph = area_graph
        self.to_vote = to_vote

        for i in range(self.agent_classes):
            agent = Actor(area_graph, in_dim, hidden_dim, out_dim).to(device)
            critic = Critic(20, hidden_dim, 1).to(device)
            buffer = ReplayBuffer(10000)
            optimizer_actor = optim.Adam(agent.parameters(), lr=0.1)
            optimizer_critic = optim.Adam(critic.parameters(), lr=0.01)
            self.agents.append(agent)
            self.critics.append(critic)
            self.buffers.append(buffer)
            self.optimizers_actor.append(optimizer_actor)
            self.optimizers_critic.append(optimizer_critic)
        
        self.gamma = 0.99

    def push_to_buffer(self, agent_class, state, action, reward, next_state, observation):
        for i in range(len(agent_class)):
            self.buffers[agent_class[i]].push(state[i], action[i], reward[i], next_state[i], observation[i])

    def update_network(self, agent_class, batch_size, explore = False):
        if len(self.buffers[agent_class]) < batch_size:
            return
        
        transitions = self.buffers[agent_class].sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_observation = zip(*transitions)

        batch_state = [state.squeeze(0) for state in batch_state]
        batch_state = torch.stack(batch_state)
        batch_action = [batch_action_i.tolist() for batch_action_i in batch_action]
        batch_action = torch.tensor(batch_action)
        batch_reward = torch.tensor(list(batch_reward)).to(device)
        batch_next_state = [state.squeeze(0) for state in batch_next_state]
        batch_next_state = torch.stack(batch_next_state)
        
        self.optimizers_critic[agent_class].zero_grad()
        critic_reward = self.critics[agent_class](batch_observation, batch = True)
        loss_critic = torch.nn.functional.mse_loss(critic_reward, batch_reward.float())
        loss_critic.backward()
        self.optimizers_critic[agent_class].step()

        self.optimizers_actor[agent_class].zero_grad()
        batch_observation_copy = copy.deepcopy(batch_observation)
        q_values = self.agents[agent_class](batch_state, batch = True, explore = explore)
        start = 0
        for g in batch_observation_copy:
            end = start + g.number_of_nodes()
            g.ndata['features'][:, 14:19] = q_values[start:end]
            start = end

        critic_reward_actor = self.critics[agent_class](batch_observation_copy, batch = True)
        loss_actor = -torch.mean(critic_reward_actor)
        loss_actor.backward()
        self.optimizers_actor[agent_class].step()

        print("class: %d, loss_critic: %.6f, loss_actor: %.6f" % (agent_class, loss_critic, loss_actor))

    def vote(self, vote_results, land, only_top_down = False):
        vote_counts = torch.empty(5).fill_(0)

        for class_id, class_votes in vote_results.items():
            for vote in class_votes:
                if only_top_down:
                    if class_id == 1:
                        vote_counts[vote] += self.vote_weight[class_id]
                else:
                    vote_counts[vote] += self.vote_weight[class_id]

        final_result = torch.argmax(vote_counts,dim=0)

        # for class_id, class_votes in vote_results.items():
        #     for vote in class_votes:
        #         if only_top_down:
        #             if class_id < 1 and vote == final_result:
        #                 land.satisfied[class_id][final_result] += 1
        #         else:
        #             if class_id < 2 and vote == final_result:
        #                 land.satisfied[class_id][final_result] += 1
        #             elif vote == final_result:
        #                 land.satisfied[class_id][final_result] += 0.1


        
        return final_result