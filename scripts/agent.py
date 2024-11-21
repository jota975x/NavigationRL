
"""
conda env create -f env.yml
conda activate ppo_env
"""


import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PolicyNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.policy_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * out_dim),  # times 2 due to mu and variance
            NormalParamExtractor()  # extract mu and variance to construct the distribution
        )

    def forward(self, x):
        return self.policy_net(x)

class CriticNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CriticNetwork, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.q_net(x)

class PPOAgent:
    def __init__(self, env, gamma=0.99, clip=0.2, epochs=10, lr=1e-4, device=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.lambda_gae = 0.95
        self.clip = clip
        self.epochs = epochs
        self.frames_per_batch = 1000
        self.total_frames = 200000
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = PolicyNetwork(
            in_dim=env.observation_space.shape[0],
            out_dim=env.action_space.shape[0]
        )
        self.critic_net = CriticNetwork(
            in_dim=env.observation_space.shape[0],
            out_dim=env.action_space.shape[0]
        )

        # Setup policy module
        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(
                self.policy_net.policy_net,
                in_keys=["observation"],
                out_keys=["loc", "scale"]
            ),
            spec=env.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "min": env.action_spec.space.low[0],
                "max": env.action_spec.space.high[0],
            },
            return_log_prob=True
        )

        # Setup critic module
        self.critic_module = ValueOperator(
            module=self.critic_net.q_net,
            in_keys=["observation"]
        )

        # Initialize PPO loss
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.critic_module,
            clip_param=clip,
            entropy_coef=0.01,
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(self.loss_module.parameters(), lr=lr)

        # Initialize data collector
        self.collector = SyncDataCollector(
            self.env,
            self.policy_module,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            device=self.device,
        )

        # Setup replay buffer
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=100),
            sampler=SamplerWithoutReplacement(),
        )

        # Setup advantage calculation
        self.advantage_module = GAE(
            gamma=self.gamma,
            lmbda=self.lambda_gae,
            value_network=self.critic_module,
            average_gae=True
        )

    def train(self):
        loss_objective_list = []
        loss_critic_list = []
        loss_entropy_list = []
        rewards = []

        for i, tensordict_data in enumerate(self.collector):
            tensordict_data = self.advantage_module(tensordict_data)
            data = tensordict_data.reshape(-1)
            self.replay_buffer.extend(data)

            r = tensordict_data.get('next').get("reward").sum().item()
            episode_rewards = [r]

            for _ in range(self.epochs):
                subdata = self.replay_buffer.sample(64)
                loss_val = self.loss_module(subdata.to(self.device))
                
                total_loss = (
                    loss_val["loss_objective"] +
                    loss_val["loss_critic"] +
                    loss_val["loss_entropy"]
                )

                loss_objective_list.append(loss_val["loss_objective"].cpu().item())
                loss_critic_list.append(loss_val["loss_critic"].cpu().item())
                loss_entropy_list.append(loss_val["loss_entropy"].cpu().item())

                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if i % 10 == 0:
                with torch.no_grad():
                    avg_reward = sum(episode_rewards) / len(episode_rewards)
                    print(f"Episodic Reward at iteration {i}: {avg_reward}")
                    rewards.append(avg_reward)

        return loss_objective_list, loss_critic_list, loss_entropy_list, rewards
    

"""
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat
from torchrl.envs.libs.gym import GymEnv

# Create and transform the environment
env = GymEnv("InvertedDoublePendulum-v4")
env = TransformedEnv(env, Compose(DoubleToFloat(["observation"])))

# Initialize the agent
agent = PPOAgent(env=env)

# Train the agent
losses_obj, losses_critic, losses_entropy, rewards = agent.train()
"""
