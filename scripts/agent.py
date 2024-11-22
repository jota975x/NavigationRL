import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAP_FEATURE_DIM = 128
MAP_SIZE = 750
STATE_DIM = 493
ACTION_DIM = 2
LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-8
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
ACTION_LOW = torch.tensor([-1.0, -0.5]).to(device)
ACTION_HIGH = torch.tensor([1.0, 0.5]).to(device)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.action_scale = ACTION_HIGH - ACTION_LOW
        self.action_bias = ACTION_LOW
        
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Flatten(),
            nn.Linear(64 * (MAP_SIZE // 8) * (MAP_SIZE // 8), MAP_FEATURE_DIM),
            nn.ReLU()
        )
        
        self.FC = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation
        )
        
        self.mu = nn.Linear(64, action_dim)
        self.log_std = nn.Linear(64, action_dim)
        
    def forward(self, x, goal):
        maps, poses, lidars = [], [], []
        
        for i in range(len(x)):
            map_ = torch.tensor(x[i]['map'], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
            pose_ = torch.tensor(x[i]['pose'], dtype=torch.float32).to(device)
            lidar_ = torch.tensor(x[i]['lidar'], dtype=torch.float32).to(device)
            
            # get map features
            map_features = self.MapFeatureExtraction(map_).squeeze()
            
            maps.append(map_features)
            poses.append(pose_)
            lidars.append(lidar_)
            
        maps = torch.stack(maps).to(device)
        poses = torch.stack(poses).to(device)
        lidars = torch.stack(lidars).to(device)
        goal = goal.unsqueeze(0).repeat(len(x), 1)
        
        # Combine inputs and pass it through policy to get action mean and log_std
        x = torch.cat((maps, poses, lidars, goal), dim=1).to(device)
        x = self.FC(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mu, log_std
    
    def sample_action(self, state, goal):
        
        mu, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        # Sample from normal distribution 
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        
        action = (torch.tanh(action) + 1.0) / 2
        action = action * self.action_scale + self.action_bias
        
        return action.detach().cpu().numpy(), log_prob
        

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Flatten(),
            nn.Linear(64 * (MAP_SIZE // 8) * (MAP_SIZE // 8), MAP_FEATURE_DIM),
            nn.ReLU()
        )
        
        self.FC = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, 1)
        )
        
    def forward(self, x, goal, action):
        
        maps, poses, lidars = [], [], []
        
        for i in range(len(x)):
            map_ = torch.tensor(x[i]['map'], dtype=torch.float32).to(device).unsqueeze(0).unsqueeze(0)
            pose_ = torch.tensor(x[i]['pose'], dtype=torch.float32).to(device)
            lidar_ = torch.tensor(x[i]['lidar'], dtype=torch.float32).to(device)
            
            # get map features
            map_features = self.MapFeatureExtraction(map_).squeeze()
            
            maps.append(map_features)
            poses.append(pose_)
            lidars.append(lidar_)

        maps = torch.stack(maps).to(device)
        poses = torch.stack(poses).to(device)
        lidars = torch.stack(lidars).to(device)
        actions = torch.tensor(np.array(action)).to(device)
        goal = goal.unsqueeze(0).repeat(len(x), 1)
        
        # Combine inputs and pass it through policy to get action mean and log_std
        x = torch.cat((maps, poses, lidars, goal, actions), dim=1)
        q = self.FC(x)        
        
        return q
    

class PPOAgent(nn.Module):
    def __init__(self, state_dim: int=STATE_DIM, action_dim: int=ACTION_DIM, learning_rate = LEARNING_RATE,
                 gamma=GAMMA, lambda_=LAMBDA, clip_eps=CLIP_EPS, num_epochs: int=NUM_EPOCHS):
        super(PPOAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lamda = lambda_
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        
        # Define actor/critic and their optimizers
        self.actor = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
        self.critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Create target critic
        self.target_critic = CriticNetwork(state_dim=state_dim, action_dim=action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def compute_advantages(self, states, goal, actions, next_states, rewards, dones):
        values = self.critic(states, goal, actions)
        with torch.no_grad():
            next_actions, _ = self.actor.sample_action(next_states, goal)
            next_values = self.target_critic(next_states, goal, next_actions)
        
        # GAE variables
        n_steps = len(rewards)
        advantages = np.zeros(n_steps)
        gae = 0
        
        # Traverse in reverse order
        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lamda * (1 - dones[t]) * gae
            advantages[t] = gae
            
        return advantages
    
    def update(self, states, goal, actions, old_log_probs, rewards, next_states, dones):
        old_log_probs = torch.stack(old_log_probs).to(device).detach()
        
        # compute advantages
        advantages = self.compute_advantages(states, goal, actions, next_states, rewards, dones)
        advantages = torch.tensor(advantages).to(device)
        
        # compute returns
        returns = advantages + self.target_critic(states, goal, actions).squeeze().detach()
        
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.unsqueeze(-1).detach()
        
        # PPO policy optimization
        for _ in range(self.num_epochs):
            # sample actions from current policy
            new_actions, new_logprobs = [], []
            for state in states:    # assume same foal for all trajectory
                action, log_prob = self.actor.sample_action(np.array([state]), goal)
                new_actions.append(action.squeeze())
                new_logprobs.append(log_prob.squeeze())
            new_logprobs = torch.stack(new_logprobs).to(device)
            new_actions = torch.tensor(np.array(new_actions)).to(device)
            
            # calculate ratios
            ratios = torch.exp(new_logprobs - old_log_probs)
            
            # compute surrogate losses
            obj_clip = ratios * advantages
            obj_surrogate = torch.min(obj_clip, torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages)
            policy_loss = -torch.mean(obj_surrogate)
            
            # update actor
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            
            # compute value loss (critic loss)
            critic_loss = torch.mean((returns - self.critic(states, goal, actions))**2)
            
            # update critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            