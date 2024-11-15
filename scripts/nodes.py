#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

#TODO: add logic (preprocessing to position and lidar data)

class lidarNode(Node):
    def __init__(self):
        super().__init__('lidarNode')
        
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info('Created LidarScanner Subsciptor')
        self.latest_scan = None
        
    
    def scan_callback(self, msg):
        self.latest_scan = msg
            
            
class velNode(Node):
    def __init__(self):
        super().__init__('velNode')
        
        self.velPub = self.create_publisher(Twist, '/diff_cont/cmd_vel_unstamped', 10)
        self.get_logger().info('Created Velocity Command Publisher')
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.command = None
        
        self.subscription = self.create_subscription(Twist, '/diff_cont/cmd_vel_unstamped', self.vel_callback, 10)
        self.get_logger().info('Created Velocity Command Subscriber')
        self.latest_vel = None
    
    # Publisher functions
    def timer_callback(self):
        if self.command:
            self.velPub.publish(self.command)
        
    def step(self, linear=0.0, angular=0.0):
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        
        self.command = command
        
    # Subscription functions
    def vel_callback(self, msg):
        self.latest_vel = msg
        

class posNode(Node):
    def __init__(self):
        super().__init__('posNode')
        
        self.subscription = self.create_subscription(Odometry, '/diff_cont/odom', self.pos_callback, 10)
        self.get_logger().info('Created Robot Position Subscriber')
        self.latest_pos = None
        
    def pos_callback(self, msg):
        self.latest_pos = msg.pose.pose
        
        