#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nodes import velNode, posNode, SlamIntegrationNode
import time


# HYPERPARAMETERS
action_time = 0.5


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

    # TRAINING LOOP
    while rclpy.ok():

        # TODO logic for SLAM

        # TODO logic for selecting action (including deriving the state (map + position + lidar))
        state = slam_node.get_drl_state()  # Retrieve the combined map, pose, and LIDAR state
        # take action for action time
        start_time = time.time()
        vel_node.step(linear=0.0, angular=0.0)  # replace with agent action
        while time.time() - start_time < action_time:
            rclpy.spin_once(slam_node)
            rclpy.spin_once(vel_node)
            rclpy.spin_once(pos_node)

        # TODO logic for deriving reward (should be inside agent)

        # TODO logic for agent update (should be inside agent)

    # shutdown ros2 communication
    rclpy.shutdown()


# SUPER IMPORTANT DO NOT DELETE
if __name__ == '__main__':
    main()
