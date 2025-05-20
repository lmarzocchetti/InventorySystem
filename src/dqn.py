import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def calculate_epsilon_decay_linear(epsilon_start, epsilon_end, steps=10_000_000):
    return (epsilon_end / epsilon_start) ** (1 / steps)

def epsilon_decay_exp(step, epsilon_start=1.0, epsilon_end=0.01, total_steps=10_000_000):
    decay_rate = (epsilon_end / epsilon_start) ** (1 / total_steps)
    return epsilon_start * (decay_rate ** step)

EPSILON_START = 1.0
EPSILON_END = 0.01
# TODO: non usare il decay ma moltiplicare l'epsilon corrente per un valore cosi che al massimo ti time-step converga a 0.01: FATTO(?)
# EPSILON_DECAY = calculate_epsilon_decay_linear(EPSILON_START, EPSILON_END)
# EPSILON_DECAY = 0.99

# EPSILON_DECAY = 0.98 # Funziona riprovare

EPSILON_DECAY = calculate_epsilon_decay_linear(EPSILON_START, EPSILON_END, steps=2_000_000) - 0.00001

# Metto alto cosi va piu veloce
TARGET_UPDATE = 5_000
GAMMA = 0.99

# Prova tanh e fare rete meno profonda
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim = 128):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)  # Output = 26 * 26 (index in a matrix with all pairs)
        )
        #TODO: .eval solo su target network: FATTO

    def forward(self, x):
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, next_state):
        if state is None or actions is None or reward is None or next_state is None:
            return
        self.buffer.append((state, actions, reward, next_state))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)        
        return (
            torch.tensor(np.array(states), dtype=torch.float).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float).to(device),
            torch.tensor(np.array(next_states), dtype=torch.float).to(device)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, device: torch.device, lr: float = 0.001, memory_size: int = 20000, hidden_dim: int = 128):
        self.state_dim = state_dim
        self.action_dim = action_dim + 1
        self.output_dim = self.action_dim * self.action_dim
        
        self.device = device
        
        self.dqn: nn.Module = DQN(state_dim, self.output_dim, hidden_dim).to(device)  # Rete per il primo valore
        
        # TODO: Freezzare le target network: Fatto
        self.target_dqn: nn.Module = DQN(state_dim, self.output_dim, hidden_dim).to(device).eval()
        self.target_dqn.load_state_dict(self.dqn.state_dict())

        self.optimizer: optim.Adam = optim.Adam(self.dqn.parameters(), lr=lr)

        self.memory: ReplayBuffer = ReplayBuffer(memory_size)
        self.epsilon: float = EPSILON_START
        self.steps: int = 0
        
        self.loss_fn_values = []
        
        self.indexing_matrix: list[tuple[int, int]] = DQNAgent.precompute_actions_pairs_list(self.action_dim)
    
    @staticmethod
    def precompute_actions_pairs_list(action_dim: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(action_dim) for j in range(action_dim)]
    
    def save_model(self, path):
        torch.save({
            'dqn': self.dqn.state_dict(),
            'target_dqn': self.target_dqn.state_dict(),
            'opt': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        saved = torch.load(path)
        self.dqn.load_state_dict(saved['dqn'])
        self.target_dqn.load_state_dict(saved['target_dqn'])
        self.optimizer.load_state_dict(saved['opt'])
    
    def select_action(self, state, inference=False) -> tuple[int, int]:
        if not inference and (self.epsilon is None or random.random() < self.epsilon):
            index = random.randint(0, self.output_dim - 1)
            return self.indexing_matrix[index], index
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.dqn(state)
            index = torch.argmax(q_values).detach().cpu().item() # TODO: ho l'indice devo fare indexing nella matrice: Fatto
            return self.indexing_matrix[index], index

    def train_step(self, batch_size = 128):
        if len(self.memory) < batch_size:
            return

        # Campiona batch dal replay buffer
        # TODO: Rimosso dones: Fatto
        states, actions, rewards, next_states = self.memory.sample(batch_size, self.device)
        
        # Predizioni Q attuali
        q_values = self.dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # q_values = self.dqn(states).gather(1, self.indexing_matrix.index(actions))

        # Q target
        with torch.no_grad():
            next_q_values = self.target_dqn(next_states).max(1)[0]
            targets = rewards + GAMMA * next_q_values

        # Calcola la loss
        loss = nn.functional.mse_loss(targets, q_values)
        # loss = nn.functional.mse_loss(q_values, targets) # Funziona al contrario
        self.loss_fn_values.append(loss.item())

        # Aggiorna i pesi
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decadimento epsilon
        self.steps += 1
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        
        # Aggiornamento rete target
        if self.steps % TARGET_UPDATE == 0:
            self.target_dqn.load_state_dict(self.dqn.state_dict())
