import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

class ReinforceAgent:
    def __init__(self, input_dim: int, action_dim: int, device: torch.device, hidden_dim: int = 512, gamma: float = 0.99):
        self.input_dim = input_dim
        self.action_dim = action_dim + 1
        self.output_dim = self.action_dim * self.action_dim
        self.gamma = gamma
        self.device = device
        
        self.loss_fn_values = []
        
        self.policy_net = PolicyNet(self.input_dim, self.output_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        
        self.indexing_matrix: list[tuple[int, int]] = ReinforceAgent.precompute_actions_pairs_list(self.action_dim)
        
    @staticmethod
    def precompute_actions_pairs_list(action_dim: int) -> list[tuple[int, int]]:
        return [(i, j) for i in range(action_dim) for j in range(action_dim)]
    
    def save_model(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'opt': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        saved = torch.load(path)
        self.policy_net.load_state_dict(saved['policy_net'])
        self.optimizer.load_state_dict(saved['opt'])
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        # print(f"DEBUG-State: {state.shape}")
        probs = self.policy_net(state).to(self.device)
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        return action.item(), log_prob
    
    def index_to_action(self, index):
        return self.indexing_matrix[index]
    
    def train_step(self, rewards, log_probs):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float).to(self.device)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # print("Log probs:", [lp.item() for lp in log_probs])
        # print("Returns:", returns.tolist())
        
        # loss = -torch.sum(torch.stack(log_probs).to(self.device) * returns)
        loss = torch.sum(torch.stack(log_probs).to(self.device) * returns)
        self.loss_fn_values.append(loss.cpu().item())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()