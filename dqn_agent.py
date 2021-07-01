# Import dependencies
import numpy as np
import random

from collections import namedtuple, deque, defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

# Importing classes from other py files
from model import model
from replay import ExperienceReplay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, a, e, beta, double, prioritized, model_update, learning_rate):
    
        # Hyperparameters
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = seed
        self.e = e
        self.beta = beta
        self.a = a
        self.prioritized = prioritized
        self.double = double
        self.model_update = model_update

        # Q-Network
        self.model_local = model(state_size, action_size, seed).to(device)
        self.model_target = model(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.model_local.parameters(), lr = learning_rate)

        # Replay memory
        self.replay = ExperienceReplay(buffer_size, batch_size, self.a, prioritized)        
            
    
    def step(self, state, action, reward, next_state, done, time_step, update_value, gamma):
        
        # Experience is added to the replay buffer        
        self.replay.add(state, action, reward, next_state, done)
        
        # When enough experiences are collected in the replay buffer and every X time 
        # steps the memory is sampled and the networks are trained
        
        if (time_step % update_value == 0) and len(self.replay) > self.batch_size:
            
            # If prioritized arg is True then use prioritizef experience replay
            if self.prioritized == True:
                experiences, idx, weights = self.replay.sample(self.beta)
                self.learn_prioritized(experiences, gamma, idx, weights)
            else:
                experiences = self.replay.sample(self.beta)
                self.learn(experiences, gamma)
                
    
    def select_action(self, state, eps):
        # Getting action values based on the input state 
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        action_values = self.model_local(state)
        
        # Action is selected based on Epsilon-Greeedy
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def action(self, state, dqn):
        # Selecting action based on Greedy policy
        _, actions = dqn(state).max(dim=1, keepdim=True)
        return actions 

    def eval_action(self, states, actions, rewards, next_states, dones, gamma, dqn):
                
        Q_expected = dqn(next_states).gather(1, actions)
        q_values = rewards + (gamma * Q_expected * (1 - dones))
        
        return q_values
    
    def double_q_update(self, states, rewards, next_states, dones, gamma):
        # Using local model to choose action
        actions = self.action(next_states, self.model_local)
        # Using target model to evaluate action chosen 
        q_values = self.eval_action(states, actions, rewards, next_states, dones, gamma, self.model_target)
        
        return q_values
    
    def q_update(self, states, rewards, next_states, dones, gamma):
        
        actions = self.action(next_states, self.model_target)
        q_values = self.eval_action(states, actions, rewards, next_states, dones, gamma, self.model_target)
        
        return q_values
    
    def learn_prioritized(self, experiences, gamma, idx, weights):
      
         # Selecting whether to use double DQN or Vanilla DQN
        states, actions, rewards, next_states, dones = experiences
    
        if self.double == True:
            Q_targets = self.double_q_update(states, rewards, next_states, dones, gamma)
        else:
            Q_targets = self.q_update(states, rewards, next_states, dones, gamma)
    
        # Get expected Q values from local model
        Q_expected = self.model_local(states).gather(1, actions)
        
        # Calculating TD error and priority
        errors = abs(Q_targets - Q_expected)
        errors = errors + self.e
        priorities = errors.cpu().detach().numpy().flatten()
        
        # Updating priorities in the replay buffer with the TD error
        self.replay.update_priorities(priorities, idx)

        loss = torch.mean((errors * torch.from_numpy(weights).to(device))**2)
            
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Selecting how the update the target model        
        if self.model_update.lower() == 'soft':
            self.soft_update(self.model_local, self.model_target, 1e-3) 
        else:
            self.update_target(self.model_local, self.model_target)
        
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # Selecting whether to use double DQN or Vanilla DQN 
        if self.double == True:
            Q_targets = self.double_q_update(states, rewards, next_states, dones, gamma)
        else:
            Q_targets = self.q_update(states, rewards, next_states, dones, gamma)
    
        # Get expected Q values from local model
        Q_expected = self.model_local(states).gather(1, actions)
      
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Selecting how the update the target model
        if self.model_update.lower() == 'soft':
            self.soft_update(self.model_local, self.model_target, 1e-3) 
        else:
            self.update_target(self.model_local, self.model_target)
    
    def soft_update(self, local_model, target_model, tau):
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    def update_target(self, local_model, target_model):
       
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)





