import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from torch.distributions import Categorical, Normal
from torch.nn.utils.rnn import pad_sequence
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = np.full((capacity),None)
        self._capacity = capacity
        self._position = 0
        self._buffer_size = 0 
#         self.buffer = deque(maxlen=capacity)
    
    def push(self, transitions):
        """ transition: (obs, states, actions, joint_actions, rewards, next_states, next_joint_actions, dones, etc)
        """
        for trans in zip(*transitions):
            self.buffer[self._position] = trans
            self._position = (self._position+1) % self._capacity
            self._buffer_size = min(self._buffer_size+1,self._capacity)
#             self.buffer.append(trans)
    
    def sample(self, batch_size):
#         batch_trans = random.sample(self.buffer[:self._buffer_size], batch_size)
        idxs = np.random.choice(self._buffer_size, size=batch_size, replace=False)
        batch_trans = self.buffer[idxs]
        transitions = map(list, zip(*batch_trans))
#         transitions = map(list, zip(*random.sample(self.buffer, batch_size)))
        return transitions
    
    def clear(self):
        self.buffer = np.full((self.capacity),None)
        self._position = 0
        self._buffer_size = 0 
#         self.buffer.clear()

    def __len__(self):
        return self._buffer_size
#         return len(self.buffer)

TorchFloat = None
TorchLong = None
class Critic(nn.Module):
    def __init__(self, args, dropout_rate=0):
        """Initialization."""
        super(Critic, self).__init__()
        global TorchFloat,TorchLong
        TorchFloat = torch.cuda.FloatTensor if args.device == torch.device('cuda') else torch.FloatTensor
        TorchLong = torch.cuda.LongTensor if args.device == torch.device('cuda') else torch.LongTensor
        self.dropout = nn.Dropout(dropout_rate)
        self.N, self.action_dim = args.N, args.action_dim
        self.time_dim, self.cs_dim, self.cp_dim = 4,8,8
        self.time_emb = nn.Embedding(args.T_LEN, self.time_dim, _weight=torch.rand(args.T_LEN,self.time_dim)) # Uniform(0,1)
        self.cs_emb = nn.Embedding(args.N, self.cs_dim, _weight=torch.rand(args.N,self.cs_dim)) # Uniform(0,1)
        self.critic_net = nn.Sequential(
            nn.Linear(64, args.hiddim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hiddim, args.hiddim),
            nn.ReLU(inplace=True), 
            nn.Linear(args.hiddim, 1)
            )

        ### charging competition vector
        self.cp_linear = nn.Sequential(
            nn.Linear(30, self.cp_dim),
            nn.ReLU(inplace=True),
        )

        ### spatilly_centralized
        self.w_linear = nn.Linear(self.time_dim+self.cs_dim+self.cp_dim+6, 64)
        self.va = nn.Parameter(torch.zeros(1,64))
        nn.init.normal_(self.va.data)
        self.out_linear = nn.Sequential(
            nn.Linear(10+self.cp_dim+self.cs_dim, 64),
            nn.ReLU(inplace=True),
        )

        ### parameters initialization ###
        # for param in self.parameters():
        #     if len(param.data.shape) >= 2:
        #         orthogonal_init(param.data, gain=2**0.5)

    def forward(self, state_actions):
        time_emb = self.time_emb(state_actions[...,0].type(TorchLong))
        cs_emb = self.cs_emb(state_actions[...,1].type(TorchLong))
        cp_emb = self.cp_linear(state_actions[...,7:37])
        state_actions = torch.cat([time_emb, cs_emb, state_actions[...,2:7], state_actions[...,37:], cp_emb],dim=-1)
        scores = torch.sum(self.va*torch.tanh(self.w_linear(state_actions)),dim=-1,keepdim=True) # (B,n_spa,1)
        att_weights = torch.softmax(scores, dim=-2) # (B,n_spa,1)
        cent_state_actions = torch.sum(att_weights*state_actions,dim=-2) # (B,N,F)
        cent_state_actions = self.out_linear(cent_state_actions) #(B,N,F')
        # print(cent_state_actions.shape)
        q_values = self.critic_net(cent_state_actions)
#         if(np.random.random() > 0.999):
#             print('critic',q_values.max(),q_values.mean(),q_values.min())
        return q_values

class Actor(nn.Module):
    def __init__(self, args, dropout_rate=0):
        """Initialization."""
        super(Actor, self).__init__()
        self.N, self.action_dim = args.N, args.action_dim
        self.time_dim,self.cs_dim = 4,8
        self.time_emb = nn.Embedding(args.T_LEN, self.time_dim, _weight=torch.rand(args.T_LEN,self.time_dim)) # Uniform(0,1)
        self.cs_emb = nn.Embedding(args.N, self.cs_dim, _weight=torch.rand(args.N,self.cs_dim)) # Uniform(0,1)
        self.actor_net = nn.Sequential(
            nn.Linear(self.time_dim+self.cs_dim+5, 64), 
            nn.ReLU(inplace=True),
            nn.Linear(64, 64), # init uniform_(-stdv, stdv)
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            )

    # def load_embedding_weight(self, weight):
    #     embedding = nn.Embedding.from_pretrained(weight)
    #     return embedding

    def forward(self, observe):
        time_emb = self.time_emb(observe[...,0].type(TorchLong))
        cs_emb = self.cs_emb(observe[...,1].type(TorchLong))
        observation = torch.cat([time_emb, cs_emb, observe[...,2:]],dim=-1)
        actions = self.actor_net(observation) # (batch,N)
#         if(np.random.random() > 0.999):
#             print('actor',actions.max(),actions.mean(),actions.min())
        return actions


class OUNoise(object):
    def __init__(self, args, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0, decay_period=30):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.low, self.high = -1, 1
        self.action_dim = args.action_dim
        self.N = args.N
        
    def reset(self):
#         self.state = np.ones(self.action_dim,) * self.mu
        self.state = np.ones(self.N,) * self.mu
        
    def evolve_state(self):
        x  = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.N)
        self.state = x + dx
        return self.state
    
    def action_noise(self, action, n_iter):
        n_q, K, _ = action.shape
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, n_iter / self.decay_period)
        ou = torch.from_numpy(ou_state).type(TorchFloat)
        ou = ou.view(1,K,1).repeat(n_q,1,1)
        action = action + ou
        return action
#         return torch.clamp(action, self.low, self.high)
        