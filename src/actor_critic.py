import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        self.max_action = max_action
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return self.max_action * x  # Evita valori negativi (min ordine = 0)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim = 256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(states), 
                torch.FloatTensor(actions), 
                torch.FloatTensor(rewards).unsqueeze(1), 
                torch.FloatTensor(next_states), 
                torch.FloatTensor(dones).unsqueeze(1))
    
    def size(self):
        return len(self.buffer)

def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.1)

class DDPG:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.001, lr=1e-4):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic_target = Critic(state_dim, action_dim)
        self.max_action = max_action

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau  

        # Patching for sharing memory 
        self.action = None
        self.state = None
        self.reward = None
        self.done = False

        # self.actor.apply(init_weights)

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        # noise = noise_scale * self.max_action * np.random.randn(*action.shape) # THIS
        # action += noise * np.random.randn(*action.shape)
        # action = np.clip(action + noise, 0, self.actor.max_action) # THIS
        action = np.clip(action, 0, self.actor.max_action)  
        return action.astype(int)

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return  

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Calcolo del valore target
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions).detach()
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Aggiornamento del Critico
        q_values = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Aggiornamento dell'Attore
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update delle reti target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
