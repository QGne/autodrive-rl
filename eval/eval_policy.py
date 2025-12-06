# eval/eval_policy.py

import numpy as np
import torch

from envs.carla_lane_env import CarlaLaneEnv
from dqn.agent import DQNAgent


def evaluate_policy(
    model_path: str = "results/dqn_carla_lane.pth",
    num_episodes: int = 5,
    max_steps_per_episode: int = 500,
):
    """
    Load a trained DQN model and run greedy evaluation episodes
    with realtime visualization.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Using device: {device}")

    # Create environment
    env = CarlaLaneEnv(max_steps_per_episode=max_steps_per_episode)
    env.realtime_render = True  # show car motion at human-friendly frame rate

    state_dim = env.state_dim
    action_dim = env.action_space_n

    # Build agent and load trained weights
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_capacity=1,        # not used in eval, but needed to init
        target_update_freq=1000,
        device=device,
    )

    print(f"[Eval] Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    agent.q_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)
    agent.q_net.eval()
    agent.target_net.eval()

    episode_rewards = []
    episode_lengths = []
    lane_departures = 0
    max_step_episodes = 0

    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        reason = None

        print(f"\n[Eval] Episode {ep + 1}/{num_episodes} started.")

        while not done and steps < max_steps_per_episode:
            # Greedy policy: epsilon = 0
            action = agent.select_action(state, epsilon=0.0)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            steps += 1
            state = next_state
            reason = info.get("reason", reason)

        print(f"[Eval] Episode {ep + 1} finished:")
        print(f"       Steps:  {steps}")
        print(f"       Reward: {ep_reward:.2f}")
        print(f"       Reason: {reason}")

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

        if reason == "lane_departure":
            lane_departures += 1
        elif reason == "max_steps":
            max_step_episodes += 1

    env.close()

    # Summary metrics
    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0

    print("\n========== Evaluation Summary ==========")
    print(f"Episodes run:           {num_episodes}")
    print(f"Average reward:         {mean_reward:.2f}")
    print(f"Average episode length: {mean_length:.1f} steps")
    print(f"Lane departures:        {lane_departures}")
    print(f"Max-steps episodes:     {max_step_episodes}")
    print("========================================\n")


if __name__ == "__main__":
    # You can tweak these numbers as needed
    evaluate_policy(
        model_path="results/dqn_carla_lane.pth",
        num_episodes=5,
        max_steps_per_episode=500,
    )
