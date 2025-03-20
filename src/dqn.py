import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 10000
TARGET_UPDATE = 100
# GAMMA = 0.99
GAMMA = 0.98

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)  # Output = 201 azioni (0-200)
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
        return (torch.FloatTensor(states),
                torch.LongTensor(actions1),
                torch.LongTensor(actions2),
                torch.FloatTensor(rewards),
                torch.FloatTensor(next_states),
                torch.FloatTensor(dones))

    def __len__(self):
        return len(self.buffer)

# Agente DQN con due reti separate
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr = 0.003, memory_size = 10000):
        self.dqn1 = DQN(state_dim, action_dim)  # Rete per il primo valore
        self.dqn2 = DQN(state_dim, action_dim)  # Rete per il secondo valore
        self.target_dqn1 = DQN(state_dim, action_dim)
        self.target_dqn2 = DQN(state_dim, action_dim)
        self.target_dqn1.load_state_dict(self.dqn1.state_dict())
        self.target_dqn2.load_state_dict(self.dqn2.state_dict())

        self.optimizer1 = optim.Adam(self.dqn1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.dqn2.parameters(), lr=lr)

        self.memory = ReplayBuffer(memory_size)
        self.epsilon = EPSILON_START
        self.steps = 0

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 200), random.randint(0, 200)
        else:
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values1 = self.dqn1(state)
                q_values2 = self.dqn2(state)
            return torch.argmax(q_values1).item(), torch.argmax(q_values2).item()

    def train_step(self, batch_size = 64):
        if len(self.memory) < batch_size:
            return

        # Campiona batch dal replay buffer
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
        loss1 = nn.MSELoss()(q_values1, targets1)
        loss2 = nn.MSELoss()(q_values2, targets2)

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

        # Aggiornamento rete target
        if self.steps % TARGET_UPDATE == 0:
            self.target_dqn1.load_state_dict(self.dqn1.state_dict())
            self.target_dqn2.load_state_dict(self.dqn2.state_dict())