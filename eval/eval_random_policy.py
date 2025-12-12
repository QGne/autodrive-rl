# eval/eval_random_policy.py

import numpy as np
import random
import time


from envs.carla_lane_env import CarlaLaneEnv


def evaluate_random_policy(
    num_episodes: int = 20,
    max_steps_per_episode: int = 500,
    success_step_threshold: int = 400,
    render: bool = False,
):
    """
    Run a pure random policy in CarlaLaneEnv for comparison with DQN.
    Metrics mirror eval_policy.py.
    """
    env = CarlaLaneEnv(max_steps_per_episode=max_steps_per_episode)
    env.realtime_render = render  # usually False for baseline
    env.spectator = env.world.get_spectator()

    episode_rewards = []
    episode_lengths = []

    collisions = 0
    lane_departures = 0
    max_step_episodes = 0
    successes = 0

    for ep in range(num_episodes):
        state = env.reset()
        ep_reward = 0.0
        steps = 0
        done = False
        reason = None

        print(f"\n[Random] Episode {ep + 1}/{num_episodes} started.")
        
        HUMAN_VIEW = True    # set False again when you don’t want slow-mo
        FPS = 20             # 20 frames per second feels about right
        STEP_DELAY = 1.0 / FPS

        while not done and steps < max_steps_per_episode:
            action = random.randrange(env.action_space_n)
            next_state, reward, done, info = env.step(action)

            ep_reward += reward
            steps += 1
            state = next_state
            reason = info.get("reason", reason)
            
            if HUMAN_VIEW:
                time.sleep(STEP_DELAY)

        print(f"[Random] Episode {ep + 1} finished:")
        print(f"         Steps:  {steps}")
        print(f"         Reward: {ep_reward:.2f}")
        print(f"         Reason: {reason}")

        episode_rewards.append(ep_reward)
        episode_lengths.append(steps)

        if reason == "collision":
            collisions += 1
        elif reason == "lane_departure":
            lane_departures += 1
        elif reason == "max_steps":
            max_step_episodes += 1

        # define "success" as surviving long enough without collision
        if steps >= success_step_threshold and reason != "collision":
            successes += 1

    env.close()

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    mean_length = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    success_rate = successes / num_episodes if num_episodes > 0 else 0.0

    print("\n========== Random Policy Evaluation Summary ==========")
    print(f"Episodes run:           {num_episodes}")
    print(f"Average reward:         {mean_reward:.2f}")
    print(f"Average episode length: {mean_length:.1f} steps")
    print(f"Success rate:           {success_rate*100:.1f}% "
          f"(steps >= {success_step_threshold}, no collision)")
    print(f"Collisions:             {collisions}")
    print(f"Lane departures:        {lane_departures}")
    print(f"Max-steps episodes:     {max_step_episodes}")
    print("======================================================\n")


if __name__ == "__main__":
    evaluate_random_policy(
        num_episodes=20,
        max_steps_per_episode=500,
        success_step_threshold=400,
        render=False,
    )
