
import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import os

# Models to compare
models = {
    "baseline": "models/baseline_ppo",
    "time_penalty": "models/time_penalty_ppo",
    "stability": "models/stability_ppo"
}

num_episodes = 100
all_results = []

os.makedirs("results", exist_ok=True)

for model_name, model_path in models.items():

    print(f"\nEvaluating {model_name}...")

    model = PPO.load(model_path)

    # Original environment: no custom rewards
    env = gym.make("LunarLander-v3")

    for episode in range(num_episodes):
        obs, info = env.reset(seed=1000 + episode)

        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

        # Preliminary landing success proxy
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
            "success": int(success)
        })

    env.close()

# Save all evaluation results
df = pd.DataFrame(all_results)

df.to_csv(
    "results/model_comparison.csv",
    index=False
)

# Calculate summary statistics
summary = df.groupby("model").agg(
    average_reward=("reward", "mean"),
    success_rate=("success", "mean"),
    average_steps=("steps", "mean")
)

summary["success_rate"] *= 100

print("\n--- MODEL COMPARISON ---")
print(summary.round(2))

summary.to_csv("results/comparison_summary.csv")
