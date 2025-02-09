from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim: int = 256):
        super().__init__()
        self.device = 'cpu'
        layers = [
            nn.Linear(in_dim, hidden_dim).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(hidden_dim, hidden_dim).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(hidden_dim, out_dim).to(self.device),
        ]
        self.model = nn.Sequential(*layers).to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        action_values = self.model(x)
        return action_values

class ActionSpace():
    def __init__(self):
        pass

class ReinforcementWarehouse:
    def __init__(self, in_dim: int = 6, out_dim: int = 2):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.policy_net = DQN(in_dim, out_dim)
        self.target_net = DQN(in_dim, out_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = 0.01)
        self.memory = deque(maxlen=10_000)
        self.batch_size = 64
        self.epsilon_start = 0.9
        self.epsilon_end = 0.005
        self.num_of_episodes = 1000 # 50 days * 100
        self.epsilon_decay = self.num_of_episodes * 10
        self.target_update = 10

        self.steps_done = 0
        self.state = (60, 60, 0, 0, 0, 0)
        self.action = None

        self.epsilon = None
        self.return_value: tuple[np.ndarray, float, bool, bool] = None
    
    def load_new_dict(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def calculate_epsilon(self, steps_done):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * steps_done / self.epsilon_decay)

    def select_action(self, state, epsilon, action_space):
        """
        Function to select an action based on an epsilon-greedy policy
        """

        if np.random.random() < epsilon:
            # return action_space.sample()
            return next(action_space)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0)
                q_values: torch.Tensor = self.policy_net(state)
                return q_values.cpu().squeeze().clone().detach().numpy()

    def optimize_model(self, batch, gamma):
        """
        Function to update the Q-values using the Bellman equation
        """

        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        q_values = self.policy_net(states)# .gather(1, actions)
        # next_q_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
        next_q_values = self.target_net(next_states)
        # print(f"{next_q_values.shape}")
        target_q_values = rewards + (gamma * next_q_values * (1 - dones))
        
        loss = nn.functional.mse_loss(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()