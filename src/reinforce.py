import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class ReinforceAgent:
    def __init__(self, input_dim, action_dim, device, hidden_dim=128, 
                 gamma=0.99, policy_lr=1e-3):
        self.input_dim = input_dim
        self.action_dim = action_dim + 1
        self.output_dim = self.action_dim * self.action_dim
        self.device = device
        
        self.loss_fn_values = []
        
        self.policy = PolicyNetwork(self.input_dim, self.output_dim, hidden_dim).to(device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.gamma = gamma
        
        self.indexing_matrix: list[tuple[int, int]] = ReinforceAgent.precompute_actions_pairs_list(self.action_dim)

    @staticmethod
    def precompute_actions_pairs_list(action_dim: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(action_dim) for j in range(action_dim)]    

    def save_model(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'opt_policy': self.policy_optim.state_dict(),
        }, path)

    def load_model(self, path):
        saved = torch.load(path)
        self.policy.load_state_dict(saved['policy'])
        self.policy_optim.load_state_dict(saved['opt_policy'])
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action_probs = self.policy(state)
        action_dist = Categorical(action_probs)
        action_idx = action_dist.sample()
        action = self.indexing_matrix[action_idx]
        entropy = action_dist.entropy()
        return action, action_idx, action_dist.log_prob(action_idx), entropy
    
    def train_step(self, rewards, log_probs, entropies):
        gammas = torch.tensor([self.gamma**k for k in range(len(rewards))], dtype=torch.float)
        
        returns = []
        G = 0
        for r, disc_gamma in zip(reversed(rewards), gammas):
            G = r + disc_gamma * G
            returns.insert(0, G)
        
        # Normalize returns
        returns = torch.tensor(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
                
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
                
        policy_loss = torch.stack(policy_loss).mean()
        entropy_loss = -torch.stack(entropies).mean()
        
        total_loss = policy_loss + 0.05 * entropy_loss
        self.loss_fn_values.append(total_loss.item())
        
        # Update policy
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
