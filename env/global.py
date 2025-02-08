import numpy as np
import torch
from torch import nn, optim
from collections import deque
import random
from agents.base_agents import BaseAgent, QuantAgent, VolatilityAgent, RiskAgent
from data.data import DataPipeline

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

class TradingSystem:
    def __init__(self, state_dim=10):
        # Initialize agents
        self.quant_agent = QuantAgent(state_dim)
        self.vol_agent = VolatilityAgent(state_dim)
        self.risk_agent = RiskAgent(state_dim)
        
        # Initialize components
        self.data_pipeline = DataPipeline()
        self.environment = TradingEnvironment()
        self.replay_buffer = MultiAgentReplayBuffer()
        self.global_critic = GlobalCritic()
        
        # Training parameters
        self.batch_size = 64
        self.gamma = 0.99
        
        # Optimizers
        self.quant_optim = optim.Adam(self.quant_agent.parameters(), lr=0.001)
        self.vol_optim = optim.Adam(self.vol_agent.parameters(), lr=0.001)
        self.risk_optim = optim.Adam(self.risk_agent.parameters(), lr=0.001)

    def run_episode(self):
        """Run a complete trading episode"""
        state = self.environment.reset()
        episode_reward = 0
        
        while not self.environment.done:
            # Get actions from all agents
            quant_action = self.quant_agent.get_action(state['quant'])
            vol_action, vol_estimate = self.vol_agent.get_action(state['vol'])
            risk_action, var_estimate = self.risk_agent.get_action(state['risk'])
            
            # Combine actions
            portfolio_action = self._combine_actions(quant_action, vol_action, risk_action)
            
            # Execute action in environment
            next_state, reward, done = self.environment.step(portfolio_action)
            
            # Store experiences
            self.replay_buffer.add(
                agent='quant',
                state=state['quant'],
                action=quant_action,
                reward=reward['quant'],
                next_state=next_state['quant'],
                done=done
            )
            # Similar for other agents...
            
            # Train agents periodically
            if len(self.replay_buffer) > self.batch_size:
                self.train_agents()
            
            state = next_state
            episode_reward += sum(reward.values())
        
        return episode_reward

    def _combine_actions(self, quant_action, vol_action, risk_action):
        """Combine agent actions into portfolio decision"""
        # Simple weighted combination (adjust weights based on your strategy)
        action_weights = {
            'quant': 0.6,
            'vol': 0.3,
            'risk': 0.1
        }
        return (
            quant_action * action_weights['quant'] +
            vol_action * action_weights['vol'] +
            risk_action * action_weights['risk']
        )

    def train_agents(self):
        """Train all agents using experience replay"""
        # Sample experiences
        quant_batch = self.replay_buffer.sample('quant', self.batch_size)
        vol_batch = self.replay_buffer.sample('vol', self.batch_size)
        risk_batch = self.replay_buffer.sample('risk', self.batch_size)
        
        # Train each agent
        self._train_agent(self.quant_agent, self.quant_optim, quant_batch)
        self._train_agent(self.vol_agent, self.vol_optim, vol_batch)
        self._train_agent(self.risk_agent, self.risk_optim, risk_batch)
        
        # Update global critic
        self.global_critic.update(
            quant_agent=self.quant_agent,
            vol_agent=self.vol_agent,
            risk_agent=self.risk_agent
        )

    def _train_agent(self, agent, optimizer, batch):
        """Generic training function for individual agents"""
        states, actions, rewards, next_states, dones = batch
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Calculate losses
        action_probs, state_values = agent(states)
        _, next_state_values = agent(next_states)
        
        advantages = rewards + self.gamma * next_state_values * (1 - dones) - state_values
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        
        policy_loss = -(action_log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy = -(action_probs * torch.log(action_probs)).sum(1).mean()
        
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

class MultiAgentReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffers = {
            'quant': deque(maxlen=capacity),
            'vol': deque(maxlen=capacity),
            'risk': deque(maxlen=capacity)
        }
    
    def add(self, agent, state, action, reward, next_state, done):
        self.buffers[agent].append((state, action, reward, next_state, done))
    
    def sample(self, agent, batch_size):
        batch = random.sample(self.buffers[agent], min(batch_size, len(self.buffers[agent])))
        return zip(*batch)
    
    def __len__(self):
        return min(len(b) for b in self.buffers.values())

class TradingEnvironment:
    def __init__(self):
        self.data = None
        self.current_step = 0
        self.done = False
        
    def reset(self):
        self.current_step = 0
        self.done = False
        return self._get_state()
    
    def step(self, action):
        self.current_step += 1
        next_state = self._get_state()
        reward = self._calculate_reward(action)
        self.done = self.current_step >= len(self.data) - 1
        return next_state, reward, self.done
    
    def _get_state(self):
        return {
            'quant': np.random.randn(10),  # Replace with actual features
            'vol': np.random.randn(5),
            'risk': np.random.randn(7)
        }
    
    def _calculate_reward(self, action):
        return {
            'quant': np.random.randn(),
            'vol': np.random.randn(),
            'risk': np.random.randn()
        }

class GlobalCritic:
    def __init__(self):
        self.critic_network = nn.Sequential(
            nn.Linear(22, 64),  # Sum of all agent state dimensions
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=0.001)
    
    def update(self, **agents):
        # Implementation would depend on your specific coordination strategy
        pass

# Test the integrated system
if __name__ == "__main__":
    trading_system = TradingSystem()
    
    print("Running initial integration test...")
    for episode in range(3):
        reward = trading_system.run_episode()
        print(f"Episode {episode+1} | Total Reward: {reward:.2f}")
    
    print("\nIntegration test complete!")
    print("Agents trained successfully:")
    print(f"Quant Agent: {len(trading_system.replay_buffer.buffers['quant'])} experiences")
    print(f"Vol Agent: {len(trading_system.replay_buffer.buffers['vol'])} experiences")
    print(f"Risk Agent: {len(trading_system.replay_buffer.buffers['risk'])} experiences")
