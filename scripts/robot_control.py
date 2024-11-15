#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nodes import velNode, lidarNode

def main(args=None):
    # iniate ros2 communication
    rclpy.init(args=args)
    
    vel_node = velNode()
    lidar_node = lidarNode()
    
    while rclpy.ok():
        # spin nodes first
        rclpy.spin_once(lidar_node)
        rclpy.spin_once(vel_node)
        
        ## INSERT LOGIC HERE
        
               
    # shutdown ros2 communication
    rclpy.shutdown()

# SUPER IMPORTANT DO NOT DELETE
if __name__ == '__main__':
    main()