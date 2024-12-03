import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scripts.utils import dict_to_tensors, states_to_stack

# HYPERPARAMETERS
ACTOR_HIDDEN_DIM = 256
LOG_MIN = -20
LOG_MAX = 2
ACTION_SCALE = 1.0
ACTION_BIAS = 0.5

CRITIC_HIDDEN_DIM = 256

LEARNING_RATE = 3e-4
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.5
ALPHA_DECAY = 0.999
ALPHA_MIN = 0.05

MAP_RESOLUTION = 0.1    # each cell in the grid map is (MAP_RESOLUTION meters)
MAP_SIZE = 30   # map_size (meters)
GRID_SIZE = int(MAP_SIZE / MAP_RESOLUTION)
MAP_FEATURE_DIM = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim: int=ACTOR_HIDDEN_DIM, action_scale=ACTION_SCALE, action_bias=ACTION_BIAS):
        super(Actor, self).__init__()
        
        # variables
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # networks
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Flatten(),
            nn.Linear(32 * ((GRID_SIZE+1) // 8) * ((GRID_SIZE+1) // 8), MAP_FEATURE_DIM),
            nn.ReLU()
        )
        
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, map, lidar, robot):
        # map feature extraction
        map_features = self.MapFeatureExtraction(map)

        # policy feedforward pass
        x = torch.cat([map_features, lidar, robot], dim=1)
        x = self.policy(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_MIN, max=LOG_MAX)
        return mean, log_std
    
    def sample(self, map, lidar, robot):
        mean, log_std = self(map, lidar, robot)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        action = torch.tanh(action)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2) + 1e-6))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob
    

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim=CRITIC_HIDDEN_DIM):
        super(Critic, self).__init__()
        
        # variables
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # networks
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Flatten(),
            nn.Linear(32 * ((GRID_SIZE+1) // 8) * ((GRID_SIZE+1) // 8), MAP_FEATURE_DIM),
            nn.ReLU()
        )
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, maps, robot, lidar, action):
        # map feature extraction
        features = self.MapFeatureExtraction(maps)

        # critic forward pass
        sa = torch.cat([features, robot, lidar, action], dim=1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
    

class AgentSAC(nn.Module):
    def __init__(self, state_dim, action_dim,
                 lr=LEARNING_RATE, gamma=GAMMA, tau=TAU, 
                 alpha=ALPHA, alpha_decay=ALPHA_DECAY, alpha_min=ALPHA_MIN):
        super(AgentSAC, self).__init__()
        
        # initialize networks
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim)
        self.critic = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic_target = Critic(state_dim=state_dim, action_dim=action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # variables
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min
        
    def update_critic(self, states, actions, next_states, rewards, dones):
        states = [dict_to_tensors(state) for state in states]
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        next_states = [dict_to_tensors(next_state) for next_state in next_states]
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32).to(device).unsqueeze(-1)
        
        # reshape states and next_states
        maps_state, robot_state, lidar_state = states_to_stack(states)
        maps_next, robot_next, lidar_next = states_to_stack(next_states)
        
        # print("critic state", maps_state.shape, robot_state.shape, lidar_state.shape)
        # print("critic next", maps_next.shape, robot_next.shape, lidar_next.shape)
        
        # update critic
        with torch.no_grad():
            next_actions, next_logprobs = self.actor.sample(maps_next, robot_next, lidar_next)
            # print("critic next actions", next_actions.shape, next_logprobs.shape)
            target_q1, target_q2 = self.critic_target(maps_next, robot_next, lidar_next, next_actions)
            # print("critic targets", target_q1.shape, target_q2.shape)
            target_q = torch.min(target_q1, target_q2)
            # print("critic target", target_q.shape)
            target_q = rewards + self.gamma * (1 - dones) * (target_q - self.alpha * next_logprobs)
            # print("critic target final", target_q.shape)
            
        current_q1, current_q2 = self.critic(maps_state, robot_state, lidar_state, actions)
        # print("critic current", current_q1.shape, current_q2.shape)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        # print("critic loss", critic_loss.shape)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # target critic soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def update_actor(self, states):
        states = [dict_to_tensors(state) for state in states]
        maps_state, robot_state, lidar_state = states_to_stack(states)
        # print("actor state", maps_state.shape, robot_state.shape, lidar_state.shape)
        
        # update actor
        actions, logprobs = self.actor.sample(maps_state, robot_state, lidar_state)
        # print("actor actions", actions.shape, logprobs.shape)
        q1, q2 = self.critic(maps_state, robot_state, lidar_state, actions)
        # print("actor q", q1.shape, q2.shape)
        actor_loss = (self.alpha * logprobs - torch.min(q1, q2)).mean()
        # print("actor loss", actor_loss.shape)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # decay alpha
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
            
        
        