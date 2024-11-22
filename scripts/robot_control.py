#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nodes import velNode, posNode, SlamIntegrationNode
import time
from agent import PPOAgent
import torch
import numpy as np

# HYPERPARAMETERS
TOTAL_EPISODE_TIME = 1
ACTION_TIME = 0.2
NUM_ACTION_PER_EPISODE = int(TOTAL_EPISODE_TIME // ACTION_TIME)
NUM_EPISODES = 1
NEW_GOAL_EVERY = 5

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create agent
agent = PPOAgent().to(device)


def main(args=None):
    # iniate ros2 communication
    rclpy.init(args=args)

    # init nodes
    slam_node = SlamIntegrationNode()
    vel_node = velNode()
    pos_node = posNode()
    rclpy.spin_once(vel_node)
    rclpy.spin_once(pos_node)
    rclpy.spin_once(slam_node)

    episode_rewards = []

    # TRAINING LOOP
    for episode in range(NUM_EPISODES):
        
        # TODO reset environment (selecting robot spawn point, select a new goal every few episodes)
        goal = torch.tensor([1, 10], dtype=torch.float32).to(device)
        
        # initialize trajectory variables
        total_reward = 0
        states, actions, old_log_probs, rewards, next_states, dones = [], [], [], [], [], []
        
        for step in range(NUM_ACTION_PER_EPISODE):
            
            state = slam_node.get_drl_state()   # retrieve combined map, pose and LiDAR state
            
            # select action and perform for ACTION_TIME
            action, logprob = agent.actor.sample_action(np.array([state]), goal)
            start_time = time.time()
            vel_node.step(linear=float(action.squeeze()[0]), angular=float(action.squeeze()[1]))
            while time.time() - start_time < ACTION_TIME:
                rclpy.spin_once(slam_node)
                rclpy.spin_once(vel_node)
                rclpy.spin_once(pos_node)
                
            next_state = slam_node.get_drl_state()
            
            # TODO logic for deriving reward
            reward = 1
            
            # TODO logic for deriving if done
            done = False
            
            # save trajectory
            states.append(state)
            actions.append(action.squeeze())
            old_log_probs.append(logprob.squeeze())
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            # Break if episode if done
            if done:
                break
            
        # Agent Update with Rollout (trajectory)
        agent.update(states, goal, old_log_probs, rewards, next_states, dones)
        
        # TODO Log episode metrics
        episode_rewards.append(total_reward)
    
    # shutdown ros2 communication
    rclpy.shutdown()


# SUPER IMPORTANT DO NOT DELETE
if __name__ == '__main__':
    main()
