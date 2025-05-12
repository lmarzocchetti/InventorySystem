import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# EPSILON_START = 1.0
# EPSILON_END = 0.05
# EPSILON_DECAY = 10000
# TARGET_UPDATE = 100
# GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
TARGET_UPDATE = 1000
GAMMA = 0.99

episodes_since_improvement = 0
best_reward = 0
RESET_THRESHOLD = 30  # episodes with no improvement
RESET_EPSILON = False # toggle reset behavior

def get_epsilon(steps):
    EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1.0 * steps / EPSILON_DECAY)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # Output = 201 azioni (0-200)
        )

    def forward(self, x):
        return self.model(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action1, action2, reward, next_state, done):
        self.buffer.append((state, action1, action2, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions1, actions2, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)),
                torch.LongTensor(actions1),
                torch.LongTensor(actions2),
                torch.FloatTensor(rewards),
                torch.FloatTensor(np.array(next_states)),
                torch.FloatTensor(torch.tensor(dones, dtype=torch.float)))

    def __len__(self):
        return len(self.buffer)

# Agente DQN con due reti separate
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr = 0.0001, memory_size = 20000, hidden_dim = 1024):
        self.dqn1 = DQN(state_dim, action_dim, hidden_dim)  # Rete per il primo valore
        self.dqn2 = DQN(state_dim, action_dim, hidden_dim)  # Rete per il secondo valore
        self.target_dqn1 = DQN(state_dim, action_dim, hidden_dim)
        self.target_dqn2 = DQN(state_dim, action_dim, hidden_dim)
        self.target_dqn1.load_state_dict(self.dqn1.state_dict())
        self.target_dqn2.load_state_dict(self.dqn2.state_dict())

        self.optimizer1 = optim.Adam(self.dqn1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.dqn2.parameters(), lr=lr)
        # self.optimizer1 = optim.SGD(self.dqn1.parameters())
        # self.optimizer2 = optim.SGD(self.dqn2.parameters())

        self.memory = ReplayBuffer(memory_size)
        self.epsilon = EPSILON_START
        self.steps = 0
    
    def save_model(self, path):
        torch.save({
            'dqn1': self.dqn1.state_dict(),
            'dqn2': self.dqn2.state_dict(),
            'target_dqn1': self.target_dqn1.state_dict(),
            'target_dqn2': self.target_dqn2.state_dict(),
            'opt1': self.optimizer1.state_dict(),
            'opt2': self.optimizer2.state_dict()
        }, path)

    def load_model(self, path):
        saved = torch.load(path)
        self.dqn1.load_state_dict(saved['dqn1'])
        self.dqn2.load_state_dict(saved['dqn2'])
        self.target_dqn1.load_state_dict(saved['target_dqn1'])
        self.target_dqn2.load_state_dict(saved['target_dqn2'])
        self.optimizer1.load_state_dict(saved['opt1'])
        self.optimizer2.load_state_dict(saved['opt2'])

    # def select_action(self, state):
    #     # if self.epsilon is None or random.random() < self.epsilon:
    #     #     return random.randint(0, 200), random.randint(0, 200)
    #     # else:
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     with torch.no_grad():
    #         q_values1 = self.dqn1(state)
    #         q_values2 = self.dqn2(state)
    #     return torch.argmax(q_values1).item(), torch.argmax(q_values2).item()
    
    def select_action(self, state):
        if self.epsilon is None or random.random() < self.epsilon:
            return random.randint(0, 200), random.randint(0, 200)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values1 = self.dqn1(state)
                q_values2 = self.dqn2(state)
            return torch.argmax(q_values1).item(), torch.argmax(q_values2).item()

    def train_step(self, total_reward, batch_size = 64):
        global episodes_since_improvement
        global best_reward
        if len(self.memory) < batch_size:
            return

        # Campiona batch dal replay buffer
        # TODO: Rimosso dones
        states, actions1, actions2, rewards, next_states, dones = self.memory.sample(batch_size)

        # Predizioni Q attuali
        q_values1 = self.dqn1(states).gather(1, actions1.unsqueeze(1)).squeeze(1)
        q_values2 = self.dqn2(states).gather(1, actions2.unsqueeze(1)).squeeze(1)

        # Q target
        with torch.no_grad():
            next_q_values1 = self.target_dqn1(next_states).max(1)[0]
            next_q_values2 = self.target_dqn2(next_states).max(1)[0]
            targets1 = rewards + GAMMA * next_q_values1 * (1 - dones)
            targets2 = rewards + GAMMA * next_q_values2 * (1 - dones)

        # Calcola la loss
        # loss1 = nn.MSELoss()(q_values1, targets1)
        # loss2 = nn.MSELoss()(q_values2, targets2)
        loss1 = nn.functional.mse_loss(targets1, q_values1)
        loss2 = nn.functional.mse_loss(targets2, q_values2)

        # Aggiorna i pesi
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()

        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()

        # Decadimento epsilon
        self.steps += 1
        self.epsilon = max(EPSILON_END, EPSILON_START - (self.steps / EPSILON_DECAY))
        # self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * torch.exp(-self.steps / EPSILON_DECAY)
        # Update epsilon
        # if RESET_EPSILON and episodes_since_improvement >= RESET_THRESHOLD:
        #     print("Resetting epsilon due to no improvement.", flush=True)
        #     self.epsilon = EPSILON_START
        #     episodes_since_improvement = 0
        # else:
        #     self.epsilon = get_epsilon(self.steps)

        # # Track performance improvement
        # if total_reward > best_reward:
        #     best_reward = total_reward
        #     episodes_since_improvement = 0
        # else:
        #     episodes_since_improvement += 1
        
        # Aggiornamento rete target
        if self.steps % TARGET_UPDATE == 0:
            self.target_dqn1.load_state_dict(self.dqn1.state_dict())
            self.target_dqn2.load_state_dict(self.dqn2.state_dict())
