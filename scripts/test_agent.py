import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from torchrl.envs import TransformedEnv, Compose, DoubleToFloat
from torchrl.envs.libs.gym import GymEnv
from agent import PPOAgent

def test_initialization():
    """Test if the agent can be properly initialized"""
    try:
        # Create and transform the environment
        env = GymEnv("InvertedDoublePendulum-v4")
        env = TransformedEnv(env, Compose(DoubleToFloat(["observation"])))
        
        # Initialize agent
        agent = PPOAgent(env=env)
        print("✓ Agent initialization successful")
        return agent, env
    except Exception as e:
        print("✗ Agent initialization failed:")
        print(f"Error: {str(e)}")
        return None, None

def test_single_training_iteration(agent):
    """Test if the agent can perform a single training iteration"""
    try:
        # Get one batch of data
        tensordict_data = next(iter(agent.collector))
        
        # Process the batch
        tensordict_data = agent.advantage_module(tensordict_data)
        data = tensordict_data.reshape(-1)
        agent.replay_buffer.extend(data)
        
        # Perform one optimization step
        subdata = agent.replay_buffer.sample(64)
        loss_val = agent.loss_module(subdata.to(agent.device))
        
        total_loss = (
            loss_val["loss_objective"] +
            loss_val["loss_critic"] +
            loss_val["loss_entropy"]
        )
        
        total_loss.backward()
        agent.optimizer.step()
        agent.optimizer.zero_grad()
        
        print("✓ Single training iteration successful")
        print(f"  - Loss objective: {loss_val['loss_objective'].item():.4f}")
        print(f"  - Loss critic: {loss_val['loss_critic'].item():.4f}")
        print(f"  - Loss entropy: {loss_val['loss_entropy'].item():.4f}")
        return True
    except Exception as e:
        print("✗ Training iteration failed:")
        print(f"Error: {str(e)}")
        return False

def test_short_training(agent, num_iterations=5):
    """Test if the agent can train for multiple iterations"""
    try:
        losses_obj, losses_critic, losses_entropy, rewards = agent.train()
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(losses_obj, label='Policy Loss')
        plt.plot(losses_critic, label='Value Loss')
        plt.plot(losses_entropy, label='Entropy Loss')
        plt.title('Training Losses')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(rewards, label='Episode Rewards')
        plt.title('Training Rewards')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.close()
        
        print("✓ Short training successful")
        print(f"  - Final average reward: {rewards[-1]:.2f}")
        return True
    except Exception as e:
        print("✗ Short training failed:")
        print(f"Error: {str(e)}")
        return False

def main():
    print("\n=== Testing PPO Agent ===\n")
    
    # Test 1: Initialization
    print("1. Testing agent initialization...")
    agent, env = test_initialization()
    if agent is None:
        return
    
    # Test 2: Single training iteration
    print("\n2. Testing single training iteration...")
    if not test_single_training_iteration(agent):
        return
    
    # Test 3: Short training run
    print("\n3. Testing short training run...")
    if not test_short_training(agent):
        return
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main() 