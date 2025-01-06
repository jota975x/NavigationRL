import random
from collections import deque, namedtuple
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward', 'done')
)

class ReplayBuffer(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        states, actions, next_states, rewards, dones = zip(*random.sample(self.memory, self.batch_size))
        
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        dones = np.array(dones)

        return states, actions, next_states, rewards, dones

    def __len__(self):
        return len(self.memory)
    
    def clear(self):
        self.memory = deque([], maxlen = self.capacity)
        return self