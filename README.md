# NavigationRL

This project explores the integration of Simultaneous Localization and Mapping (SLAM) with Deep Reinforcement Learning (DRL) to enable autonomous robot navigation in unknown environments. The approach combines proprioceptive and exteroceptive data to create a comprehensive state representation for the DRL agent, incorporating SLAM-derived localization and mapping information.

The SLAM system generates an occupancy grid map, providing a spatial representation of the environment, while odometry is used for accurate localization. These data sources, along with the robot's proprioceptive inputs, form the state space for the DRL agent. A Soft Actor-Critic (SAC) model is employed as the DRL agent, leveraging its ability to handle continuous action spaces effectively.

Simulations were conducted using Pybullet, a physics-based simulator, with Python as the development language. The robot learns to navigate dynamically in newly encountered spaces, efficiently avoiding obstacles and optimizing its path. This project demonstrates the potential of combining SLAM and DRL for intelligent, adaptive navigation in robotics.

Key Technologies and Methods:
- SLAM: Occupancy grid mapping and odometry-based localization.
- Deep Reinforcement Learning: Soft Actor-Critic (SAC) algorithm for continuous control.
- Simulation Environment: Pybullet for physics-based simulation.
- Development Language: Python.
This work highlights the synergy between SLAM and DRL, paving the way for robust autonomous navigation in real-world robotics applications.

Simulation can be performed by running `AutoNav.ipynb`. DRL model definition, LiDAR data collection, SLAM specifications and scripts, and other functions can be found on the `saves` folder.
