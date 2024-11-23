#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nodes import velNode, posNode, SlamIntegrationNode, ResetWorldClient
import time
from agent import PPOAgent
import torch
import numpy as np
from utils import get_goal, get_reward
import math

# HYPERPARAMETERS
TOTAL_EPISODE_TIME = 90
ACTION_TIME = 0.2
NUM_ACTION_PER_EPISODE = int(TOTAL_EPISODE_TIME // ACTION_TIME)
NUM_EPISODES = 500
NEW_GOAL_EVERY = 5
SAVE_MODEL_EVERY = 10

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
    reset_client = ResetWorldClient()
    rclpy.spin_once(vel_node)
    rclpy.spin_once(pos_node)
    rclpy.spin_once(slam_node)

    episode_rewards = []
    dist_to_goal_tracking = []
    
    # Avoids bugs with map and robot pose - DO NOT DELETE!!!
    for _ in range(20):
        state = slam_node.get_drl_state()
        rclpy.spin_once(slam_node)

    # TRAINING LOOP
    print("begin training")
    for episode in range(NUM_EPISODES):
        # reset world
        reset_client.reset_world()

        # change goal
        if episode % NEW_GOAL_EVERY == 0:
            goal = get_goal().to(device)

        # initialize trajectory variables
        total_reward = 0
        states, actions, old_log_probs, rewards, next_states, dones = [], [], [], [], [], []
        done = False
        
        for step in range(NUM_ACTION_PER_EPISODE):

            state = slam_node.get_drl_state()  # retrieve combined map, pose and LiDAR state

            # select action and perform for ACTION_TIME
            action, logprob, _ = agent.actor.sample_action(np.array([state]), goal)
            start_time = time.time()
            vel_node.step(linear=float(action.squeeze()[0]), angular=float(action.squeeze()[1]))
            while time.time() - start_time < ACTION_TIME:
                rclpy.spin_once(slam_node)
                rclpy.spin_once(vel_node)
                rclpy.spin_once(pos_node)

            next_state = slam_node.get_drl_state()

            # check if done and give reward
            reward = 0.0
            if (slam_node.is_wall_close()):
                done = True
                reward = -100
            if (slam_node.is_goal_reached(goal, next_state['pose'])):
                done = True
                reward = 100
            reward += get_reward(next_state['pose'], goal)
        
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
        print("Begin Update")
        agent.update(states, goal, old_log_probs, rewards, next_states, dones)
        print("Model Updated")
        
        # check final the distance to goal
        dist_goal = math.sqrt((next_state['pose'][0] - goal[0])**2 + (next_state['pose'][1] - goal[1])**2)

        # Log episode metrics
        dist_to_goal_tracking.append(dist_goal)
        episode_rewards.append(total_reward)
        
        # save model
        if episode % SAVE_MODEL_EVERY == 0:
            torch.save(agent.state_dict(), f'src/navigationrl/saves/model_state_{episode}.pth')

    # Save training data
    with open('src/navigationrl/training_log/training_log.txt', 'w') as file:
        file.write(f'Dist_to_goal: {dist_to_goal_tracking} \n')
        file.write(f'Training Rewards: {episode_rewards}')
    
    # shutdown ros2 communication
    rclpy.shutdown()


# SUPER IMPORTANT DO NOT DELETE
if __name__ == '__main__':
    main()
