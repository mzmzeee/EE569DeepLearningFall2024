import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are scaled between [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Outputs a single Q-value
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatenate state and action
        return self.net(x)

# Replay Buffer to store experience tuples
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def size(self):
        return len(self.buffer)

# DDPG Algorithm
class DDPG:
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256, gamma=0.99, tau=0.005, lr=3e-4):
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau

        self.max_action = max_action
        self.replay_buffer = ReplayBuffer(max_size=1000000)

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.normal(0, 1, size=action.shape)  # Add exploration noise
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-values
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q.squeeze()

        # Compute critic loss
        current_q = self.critic(states, actions).squeeze()
        critic_loss = nn.MSELoss()(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_network(self.actor, self.actor_target)
        self._update_target_network(self.critic, self.critic_target)

    def _update_target_network(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

def train_ddpg():
    # Hyperparameters
    env_name = "BipedalWalker-v3"
    max_episodes = 1000
    max_timesteps = 1500
    batch_size = 64
    start_timesteps = 10000  # Collect experience before training starts
    noise = 0.1
    gamma = 0.99
    tau = 0.005
    lr = 3e-4

    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize DDPG agent
    agent = DDPG(state_dim, action_dim, max_action, gamma=gamma, tau=tau, lr=lr)

    # Training loop
    total_timesteps = 0
    rewards_history = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            total_timesteps += 1

            # Select action
            if total_timesteps < start_timesteps:
                action = env.action_space.sample()  # Explore with random actions
            else:
                action = agent.select_action(state, noise=noise)

            # Take action in the environment
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated

            # Store transition in replay buffer
            agent.replay_buffer.add(state, action, reward, next_state, float(terminated))

            state = next_state
            episode_reward += reward

            # Train the agent
            if total_timesteps >= start_timesteps:
                agent.train(batch_size)

            if terminated:
                break

        rewards_history.append(episode_reward)
        print(f"Episode {episode} - Reward: {episode_reward:.2f}")

        # Stop training if the agent consistently performs well
        if len(rewards_history) >= 100 and np.mean(rewards_history[-100:]) >= 300:
            print(f"Solved in {episode} episodes!")
            break

        if episode % 50 == 0:
            env.close()
            test_ddpg(env_name, agent)
            env = gym.make(env_name)

    # Save the trained model
    torch.save(agent.actor.state_dict(), "ddpg_actor_bipedalwalker.pth")
    torch.save(agent.critic.state_dict(), "ddpg_critic_bipedalwalker.pth")
    print("Training complete. Model saved.")

    # Test the trained model
    test_ddpg(env, agent)

def test_ddpg(env_name, agent):
    env = gym.make(env_name,render_mode="human")
    print("Testing the trained model...")
    state, _ = env.reset()
    total_reward = 0

    for _ in range(1500):  # Max timesteps per episode
        env.render()
        action = agent.select_action(state, noise=0)  # No noise during testing
        state, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break

    print(f"Test Reward: {total_reward:.2f}")
    env.close()

if __name__ == "__main__":
    train_ddpg()
