import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        logits = self.net(x)
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs)

class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(input_dim, hidden_dim),
            nn.Linear(input_dim + 1, hidden_dim),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action: int, action_size: int):
        # return self.net(state).squeeze(-1)
        # a -> index
        a = torch.tensor(action, dtype=torch.float).unsqueeze(-1).unsqueeze(-1) / (action_size - 1)
        sa = torch.cat([state, a], dim=-1)
        return self.net(sa).squeeze(-1)

class ActorCriticAgent:
    def __init__(self, input_dim: int, action_dim: int, device: torch.device, hidden_dim: int = 512, gamma: float = 0.99):
        self.input_dim = input_dim
        self.action_dim = action_dim + 1
        self.output_dim = self.action_dim * self.action_dim
        self.gamma = gamma
        self.device = device
        
        self.loss_fn_actor_values = []
        self.loss_fn_critic_values = []
        
        self.actor = Actor(self.input_dim, self.output_dim, hidden_dim).to(device)
        self.critic = Critic(self.input_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.indexing_matrix: list[tuple[int, int]] = ActorCriticAgent.precompute_actions_pairs_list(self.action_dim)
        
    @staticmethod
    def precompute_actions_pairs_list(action_dim: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(action_dim) for j in range(action_dim)]
    
    def save_model(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opt_actor': self.actor_optimizer.state_dict(),
            'opt_critic': self.critic_optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        saved = torch.load(path)
        self.actor.load_state_dict(saved['actor'])
        self.critic.load_state_dict(saved['critic'])
        self.actor_optimizer.load_state_dict(saved['opt_actor'])
        self.critic_optimizer.load_state_dict(saved['opt_critic'])
    
    # def select_action(self, state):
    #     state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
    #     action_probs = self.actor(state)
    #     dist = Categorical(action_probs)
    #     action = dist.sample()
    #     log_prob = dist.log_prob(action)
    #     return self.index_to_action(action.item()), log_prob
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        dist = self.actor(state)
        action = dist.sample()
        # log_prob = dist.log_prob(action)
        return action.item(), self.index_to_action(action.item())
    
    def index_to_action(self, index):
        return self.indexing_matrix[index]
    
    # def train_step(self, state, next_state, reward, log_prob):
    #     state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
    #     next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
        
    #     with torch.no_grad():
    #         target_value = reward + self.gamma * self.critic(next_state) 
        
    #     # Critic loss (MSE)
    #     value = self.critic(state)
    #     critic_loss = F.mse_loss(target_value, value)
    #     self.loss_fn_critic_values.append(critic_loss.cpu().item())
        
    #     # Actor loss (policy)
    #     advantage = (target_value - value).detach()
    #     actor_loss = -log_prob * advantage
    #     self.loss_fn_actor_values.append(actor_loss.cpu().item())

    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
        
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
    
    def train_step(self, action, state, next_state, reward):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        
        # critic update
        with torch.no_grad():
            next_dist = self.actor(next_state)
            next_action = next_dist.sample().squeeze()
            q_next = self.critic(next_state, next_action, self.output_dim)
            target = reward + self.gamma * q_next
        
        q_pred = self.critic(state, action, self.output_dim)
        critic_loss = F.mse_loss(q_pred, target)
        self.loss_fn_critic_values.append(critic_loss.cpu().item())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        dist = self.actor(state)
        log_prob = dist.log_prob(action)
        # print(f"Debug: {log_prob}")
        q_val = self.critic(state, action, self.output_dim).detach()
        
        actor_loss = -log_prob * q_val
        self.loss_fn_actor_values.append(actor_loss.cpu().item())
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        