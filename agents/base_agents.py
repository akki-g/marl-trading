import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# MPS acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class BaseAgent(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(BaseAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        features = self.feature_extractor(state)
        action_probs = self.policy_head(features)
        state_value = self.value_head(features)
        return action_probs, state_value

class QuantAgent(BaseAgent):
    def __init__(self, state_dim, action_dim=3):  # 3 actions: Buy, Hold, Sell
        super(QuantAgent, self).__init__(state_dim, action_dim)
        
    def get_action(self, state):
        action_probs, _ = self.forward(state)
        action = torch.multinomial(action_probs, 1).item()
        return action

class VolatilityAgent(BaseAgent):
    def __init__(self, state_dim, action_dim=2):  # 2 actions: Increase/Decrease Volatility
        super(VolatilityAgent, self).__init__(state_dim, action_dim)
        
        # Additional layer for volatility estimation
        self.volatility_estimator = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def estimate_volatility(self, state):
        return self.volatility_estimator(state)
    
    def get_action(self, state):
        action_probs, _ = self.forward(state)
        volatility = self.estimate_volatility(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, volatility.item()

class RiskAgent(BaseAgent):
    def __init__(self, state_dim, action_dim=3):  # 3 actions: Increase/Maintain/Decrease Risk
        super(RiskAgent, self).__init__(state_dim, action_dim)
        
        # Additional layer for VaR estimation
        self.var_estimator = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def estimate_var(self, state):
        return self.var_estimator(state)
    
    def get_action(self, state):
        action_probs, _ = self.forward(state)
        var = self.estimate_var(state)
        action = torch.multinomial(action_probs, 1).item()
        return action, var.item()

# Training function
def train_agent(agent, optimizer, states, actions, rewards, next_states, done, gamma=0.99):
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    done = torch.FloatTensor(done)

    # Compute action probabilities and state values
    action_probs, state_values = agent(states)
    _, next_state_values = agent(next_states)

    # Compute advantages
    advantages = rewards + gamma * next_state_values * (1 - done) - state_values

    # Compute losses
    action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
    policy_loss = -(action_log_probs * advantages.detach()).mean()
    value_loss = advantages.pow(2).mean()
    
    # Entropy for exploration
    entropy = -(action_probs * torch.log(action_probs)).sum(1).mean()

    # Total loss
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

# Test the implementation
if __name__ == "__main__":
    state_dim = 10  # Example state dimension
    
    quant_agent = QuantAgent(state_dim)
    vol_agent = VolatilityAgent(state_dim)
    risk_agent = RiskAgent(state_dim)
    
    print("QuantAgent Architecture:")
    print(quant_agent)
    
    print("\nVolatilityAgent Architecture:")
    print(vol_agent)
    
    print("\nRiskAgent Architecture:")
    print(risk_agent)
    
    # Test forward pass
    test_state = torch.randn(1, state_dim)
    
    quant_action = quant_agent.get_action(test_state)
    vol_action, vol_estimate = vol_agent.get_action(test_state)
    risk_action, var_estimate = risk_agent.get_action(test_state)
    
    print(f"\nQuantAgent Action: {quant_action}")
    print(f"VolatilityAgent Action: {vol_action}, Volatility Estimate: {vol_estimate:.4f}")
    print(f"RiskAgent Action: {risk_action}, VaR Estimate: {var_estimate:.4f}")
    
    # Test training loop
    optimizer = optim.Adam(quant_agent.parameters(), lr=0.001)
    
    # Dummy data for training example
    states = np.random.randn(32, state_dim)
    actions = np.random.randint(0, 3, 32)
    rewards = np.random.randn(32)
    next_states = np.random.randn(32, state_dim)
    done = np.random.randint(0, 2, 32)
    
    loss = train_agent(quant_agent, optimizer, states, actions, rewards, next_states, done)
    print(f"\nTraining Loss: {loss:.4f}")
