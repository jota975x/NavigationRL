import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAP_FEATURE_DIM = 128
MAP_SIZE = 550
STATE_DIM = 493
ACTION_DIM = 2
LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
EPS = 1e-8
GAMMA = 0.99
LAMBDA = 0.95
ALPHA = 0.01
CLIP_EPS = 0.2
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4
BATCH_SIZE = 64 # should help with processing
ACTION_LOW = torch.tensor([-0.2, -0.5]).to(device)
ACTION_HIGH = torch.tensor([0.7, 0.5]).to(device)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM, batch_size: int=BATCH_SIZE):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.action_scale = ACTION_HIGH - ACTION_LOW
        self.action_bias = ACTION_LOW
        self.batch_size = batch_size
        
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Flatten(),
            nn.Linear(16 * (MAP_SIZE // 16) * (MAP_SIZE // 16), MAP_FEATURE_DIM),
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
        outputs = []
        
        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            
            # prepare batched data
            maps = torch.stack([torch.tensor(state['map'], dtype=torch.float32).to(device) for state in batch])
            maps = maps.unsqueeze(1)    # for shape [B, 1, H, W]
            poses = torch.stack([torch.tensor(state['pose'], dtype=torch.float32).to(device) for state in batch])
            lidars = torch.stack([torch.tensor(state['lidar'], dtype=torch.float32).to(device) for state in batch])
            
            # get map features
            map_features = self.MapFeatureExtraction(maps)
            
            # repeat goal for the current batch
            goal_batch = goal.unsqueeze(0).repeat(len(batch), 1)
            
            # combine inputs and pass through policy network
            inputs = torch.cat((map_features, poses, lidars, goal_batch), dim=1).to(device)
            policy_out = self.FC(inputs)
            mu = self.mu(policy_out)
            log_std = self.log_std(policy_out)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            
            # collect outputs
            outputs.append((mu, log_std))
            
        # concatenate all batch outputs
        mus, log_stds = zip(*outputs)
        mus = torch.cat(mus, dim=0).to(device)
        log_stds = torch.cat(log_stds, dim=0).to(device)
        
        return mus, log_stds        
        
    
    def sample_action(self, state, goal):
        
        mu, log_std = self.forward(state, goal)
        std = log_std.exp()
        
        # Sample from normal distribution 
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean()
        
        action = (torch.tanh(action) + 1.0) / 2
        action = action * self.action_scale + self.action_bias
        
        return action.detach().cpu().numpy(), log_prob, entropy
        

# TODO switch it to estimate V instead of Q (should also require other minor changes somewhere else)
class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, batch_size: int=BATCH_SIZE):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.activation = nn.ReLU()
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.MapFeatureExtraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            self.activation,
            self.maxPool,
            
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            self.activation,
            self.maxPool,
            
            
            nn.Flatten(),
            nn.Linear(16 * (MAP_SIZE // 16) * (MAP_SIZE // 16), MAP_FEATURE_DIM),
            nn.ReLU()
        )
        
        self.FC = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            self.activation,
            nn.Linear(256, 128),
            self.activation,
            nn.Linear(128, 64),
            self.activation,
            nn.Linear(64, 1)
        )
        
    def forward(self, x, goal):
        outputs = []
        
        for i in range(0, len(x), self.batch_size):
            batch = x[i:i + self.batch_size]
            
            # prepare batched data
            maps = torch.stack([torch.tensor(state['map'], dtype=torch.float32).to(device) for state in batch])
            maps = maps.unsqueeze(1)    # for shape [B, 1, H, W]
            poses = torch.stack([torch.tensor(state['pose'], dtype=torch.float32).to(device) for state in batch])
            lidars = torch.stack([torch.tensor(state['lidar'], dtype=torch.float32).to(device) for state in batch])
            
            # get map features
            map_features = self.MapFeatureExtraction(maps)
            
            # repeat goal for the current batch
            goal_batch = goal.unsqueeze(0).repeat(len(batch), 1)
            
            # combine inputs and pass through policy network
            
            inputs = torch.cat((map_features, poses, lidars, goal_batch), dim=1).to(device)
            q =self.FC(inputs)
            outputs.append(q)
            
        q_values = torch.cat(outputs, dim=0)     
            
        return q_values
    

class PPOAgent(nn.Module):
    def __init__(self, state_dim: int=STATE_DIM, action_dim: int=ACTION_DIM, learning_rate = LEARNING_RATE,
                 gamma=GAMMA, lambda_=LAMBDA, alpha=ALPHA, clip_eps=CLIP_EPS, num_epochs: int=NUM_EPOCHS, batch_size: int=BATCH_SIZE):
        super(PPOAgent, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lamda = lambda_
        self.alpha = alpha
        self.clip_eps = clip_eps
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        # Define actor/critic and their optimizers
        self.actor = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, batch_size=batch_size)
        self.critic = CriticNetwork(state_dim=state_dim, batch_size=batch_size)
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        
    def compute_returns_and_advantages(self, states, goal, next_states, rewards, dones):
        with torch.no_grad():
            values = self.critic(states, goal)
            next_values = self.critic(next_states, goal)
        
        # Compute deltas (TD Error)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)
        deltas = rewards + self.gamma * next_values.squeeze() * (1 - dones) - values.squeeze()
        
        # Compute advantages and returns using GAE
        advantages = torch.zeros_like(rewards).to(device)
        returns = torch.zeros_like(rewards).to(device)
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            gae = deltas[t] + self.gamma * self.lamda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return returns, advantages
    
    def update(self, states, goal, old_log_probs, rewards, next_states, dones):
        old_log_probs = torch.stack(old_log_probs).to(device).detach()
        
        # compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(states, goal, next_states, rewards, dones)
        returns = returns.detach()
        
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()
        
        # PPO policy optimization
        for i in range(self.num_epochs):
            # sample actions from current policy
            new_actions, new_logprobs, new_entropy = self.actor.sample_action(states, goal)
            
            # calculate ratios
            ratios = torch.exp(new_logprobs - old_log_probs)
            
            # compute surrogate losses
            obj_clip = ratios * advantages
            obj_surrogate = torch.min(obj_clip, torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages)
            policy_loss = -torch.mean(obj_surrogate)
            # add entropy term
            policy_loss = policy_loss + self.alpha * new_entropy
            
            # update actor
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            
            # compute value loss (critic loss)
            critic_loss = torch.mean((returns - self.critic(states, goal).squeeze())**2)
            
            # update critic
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
