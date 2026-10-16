
import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import os

# Models to compare

models = {
    "baseline": "models/baseline_ppo",
    "time_penalty": "models/time_penalty_ppo",
    "stability": "models/stability_ppo",
    "stability_prime": "models/stability_prime_ppo",
    "stability_double_prime": "models/stability_double_prime_ppo"
}


num_episodes = 100
all_results = []

os.makedirs("results", exist_ok=True)

for model_name, model_path in models.items():

    print(f"\nEvaluating {model_name}...")

    model = PPO.load(model_path)

    # Evaluate using the original reward function
    env = gym.make("LunarLander-v3")

    for episode in range(num_episodes):
        obs, info = env.reset(seed=1000 + episode)

        done = False
        total_reward = 0
        steps = 0

        angular_velocities = []

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Record absolute angular velocity after each step
            angular_velocity = abs(float(obs[5]))
            angular_velocities.append(angular_velocity)

            done = terminated or truncated

        # Existing preliminary success proxy
        success = (
            terminated
            and not truncated
            and reward >= 100
        )

        all_results.append({
            "model": model_name,
            "episode": episode + 1,
            "reward": total_reward,
            "steps": steps,
            "success": int(success),
            "avg_angular_velocity": np.mean(angular_velocities),
            "max_angular_velocity": np.max(angular_velocities)
        })

    env.close()

# Save episode-level results
df = pd.DataFrame(all_results)

df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Calculate model-level averages
summary = df.groupby("model").agg(
    average_reward=("reward", "mean"),
    success_rate=("success", "mean"),
    average_steps=("steps", "mean"),
    average_angular_velocity=("avg_angular_velocity", "mean"),
    average_max_angular_velocity=("max_angular_velocity", "mean")
)

summary["success_rate"] *= 100

print("\n--- MODEL COMPARISON ---")
print(summary.round(4).to_string())

summary.to_csv("results/comparison_summary.csv")

print("\nResults saved successfully!")
