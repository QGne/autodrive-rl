# train/train_dqn.py

from offline.dpm_prefill import prefill_replay_with_dpm
import numpy as np
import torch

from envs.carla_lane_env import CarlaLaneEnv
from dqn.agent import DQNAgent


def train_dqn(
    num_episodes: int = 500,
    max_steps_per_episode: int = 500,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.98,
):
    # Create environment
    env = CarlaLaneEnv(max_steps_per_episode=max_steps_per_episode)
    env.realtime_render = False  # FAST mode for training

    state_dim = env.state_dim
    action_dim = env.action_space_n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        replay_capacity=100_000,
        target_update_freq=1000,
        device=device,
    )
    

    # ---------- PDM offline replay prefill ----------
    use_pdm_prefill = True
    if use_pdm_prefill:
        route_dir = "data/NoScenario/Route0_Rep0"
        print(f"[PDM] Pre-filling agent replay buffer from {route_dir} ...")
        num_added = prefill_replay_with_dpm(
            agent.replay,          # <--- or agent.memory if that's the name
            route_dir=route_dir,
            max_prefill=50000,
            episode_max_steps=max_steps_per_episode,
        )
        print(f"[PDM] Added {num_added} offline transitions.")
    # ------------------------------------------------

    epsilon = epsilon_start
    rewards_per_episode = []

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)

            agent.replay.push(state, action, reward, next_state, done)
            loss = agent.update()

            state = next_state
            ep_reward += reward

            if done:
                break

        rewards_per_episode.append(ep_reward)

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(
            f"Episode {ep+1}/{num_episodes} "
            f"| Reward: {ep_reward:.1f} "
            f"| Epsilon: {epsilon:.3f}"
        )

    env.close()
    return agent, rewards_per_episode


if __name__ == "__main__":
    agent, rewards = train_dqn()
    # Optionally save the trained model
    torch.save(agent.q_net.state_dict(), "results/dqn_carla_lane.pth")
    #torch.save(agent.q_net.state_dict(), "results/dqn_carla_lane_big.pth")
    print("Training finished, model saved to results/dqn_carla_lane.pth")
    #print("Training finished, model saved to results/dqn_carla_lane_big.pth")
    
    import matplotlib.pyplot as plt
    import os

    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training: Episode Rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reward_curve.png")
    print("Saved reward curve to results/reward_curve.png")
