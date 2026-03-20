import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import random
from collections import deque
import argparse

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x): return self.net(x)



GAMMA = 0.99  # Discount factor (Long-term vs short-term)
LR = 5e-4  # Learning rate
BATCH_SIZE = 64  # How many "memories" to learn from at once
MEMORY_SIZE = 10000  # Total capacity of Replay Buffer
EPS_START = 1.0  # Start by exploring 100%
EPS_END = 0.05  # End with 5% exploration
EPS_DECAY = 0.995  # How fast we stop being "random"
MODEL_PATH = "" # Model save path
def train():
    cuda = torch.cuda.is_available()
    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    env = gym.make("LunarLander-v3")
    policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())  # Sync them initially
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPS_START

    record = -1000
    for episode in range(500):
        state, _ = env.reset()
        total_reward = 0

        for t in range(1000):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state).to(device)).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                s_lst, a_lst, r_lst, ns_lst, d_lst = zip(*batch)

                s = torch.FloatTensor(np.array(s_lst)).to(device)
                a = torch.LongTensor(a_lst).unsqueeze(1).to(device)
                r = torch.FloatTensor(r_lst).to(device)
                ns = torch.FloatTensor(np.array(ns_lst)).to(device)
                d = torch.FloatTensor(d_lst).to(device)

                # Q(s,a)
                current_q = policy_net(s).gather(1, a).squeeze()
                # max Q(s', a') from target net
                next_q = target_net(ns).max(1)[0].detach()
                target_q = r + (GAMMA * next_q * (1 - d))

                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done: break

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        target_net.load_state_dict(policy_net.state_dict())

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            if total_reward > record:
                policy_net._save_to_state_dict(policy_net.state_dict(), MODEL_PATH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Solve Lunar Lander with trained AI model')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount Factor (default: 0.99)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Number of games to play (default: 1)')
    parser.add_argument('--model', type=str, default='model.pth',
                        help='Path to save model file (default: model.pth)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch Size (default: 64)z')
    parser.add_argument('--memory-size', type=int, default=10000,
                        help='replay buffer size (default: 10000)')
    parser.add_argument('--eps-start', type=float, default=1.0,
                        help='Epsilon start value (default: 1)')
    parser.add_argument('--eps-decay', type=float, default=0.995,
                        help='Epsilon decay value (default: 0.995)')
    parser.add_argument('--eps-end', type=float, default=0.05,
                        help='Epsilon end value (default: 0.05)')

    args = parser.parse_args()

    GAMMA = args.gamma
    LR = args.lr
    MODEL_PATH = args.model
    BATCH_SIZE = args.batch_size
    MEMORY_SIZE = args.memory_size
    EPS_END = args.eps_end
    EPS_DECAY = args.eps_decay
    EPS_START = args.eps_start

    train()
