import numpy as np
import random

from collections import namedtuple, deque, defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ExperienceReplay():
    def __init__(self, buffer_size, batch_size, a, prioritized):
        
        self.prioritized = prioritized
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.priorities = deque(np.zeros(buffer_size, np.float32), maxlen=buffer_size)
        self.index = 0
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.a = a
               
    def add(self, state, action, reward, next_state, done):
       
        if self.prioritized == True:
            priorityCopy = self.priorities.copy()
            priorityAddArray = list(priorityCopy)
            
            priority = max(priorityAddArray) if self.buffer else 1.0

            self.buffer.append(self.experience(state, action, reward, next_state, done))

            self.priorities.append(priority)
            self.index = min((self.index+1), self.buffer_size)
            
            del priorityAddArray
            del priorityCopy
        
        else:
            self.buffer.append(self.experience(state, action, reward, next_state, done))
            
    
    def sample(self, beta):
        
        if self.prioritized == True:
            priorityCopy = self.priorities.copy()
            prioritySampleArray = np.asarray(list(priorityCopy), dtype = float)
            
            if len(self.buffer) == self.buffer_size:
                samplePriorities = prioritySampleArray
            else:
                samplePriorities = prioritySampleArray[-self.index:]
            
            sampleProbs = (samplePriorities)**self.a / (samplePriorities**self.a).sum()
           
            idx = np.random.choice(len(self.buffer), self.batch_size, p = sampleProbs, replace = True)
            
            experiences = [self.buffer[i] for i in idx]
        
            weights = (len(self.buffer) * sampleProbs[idx])**(-beta)
            weights = weights / weights.max()
            
            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

            del prioritySampleArray
            del priorityCopy
            
            return (states, actions, rewards, next_states, dones), idx, weights
            
        else:
            idx = np.random.choice(len(self.buffer), self.batch_size)
            experiences = [self.buffer[i] for i in idx]

            states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
            return (states, actions, rewards, next_states, dones)
     
    def update_priorities(self, prior, idx):
        for i in prior:
            self.priorities.append(i)

    def __len__(self):
        return len(self.buffer)