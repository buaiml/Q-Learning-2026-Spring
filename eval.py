import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from q_learn import DQN
import argparse


def evaluate(model_path, num_episodes=10, render=True):
    render_mode = "human" if render else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DQN(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print("Error: model.pth not found. Ensure you trained and saved the model first.")
        return

    model.eval()
    all_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_t).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

    avg_reward = np.mean(all_rewards)
    print(f"\nEvaluation over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Max Reward: {np.max(all_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Lunar Lander trained AI model')
    parser.add_argument('--model', type=str, default='./models/model.pth',
                        help='Path to model file (default: model.pth)')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Num episodes to evaluate (default: 10)')
    parser.add_argument('--headless', type=bool, default=True,
                        help='Whether to render evaluation environment (default: true)')
    args = parser.parse_args()
    evaluate(args.model, num_episodes=5, render=True)