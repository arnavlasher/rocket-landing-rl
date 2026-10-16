
import gymnasium as gym
from stable_baselines3 import PPO
import pandas as pd
import numpy as np
import os

models = {
    "baseline": "models/baseline_ppo",
    "time_penalty": "models/time_penalty_ppo",
    "stability": "models/stability_ppo"
}

os.makedirs("results", exist_ok=True)

all_trajectories = []

for model_name, model_path in models.items():

    print(f"Analyzing {model_name}...")

    model = PPO.load(model_path)
    env = gym.make("LunarLander-v3")

    for episode in range(100):

        obs, info = env.reset(seed=1000 + episode)

        done = False
        step = 0

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(action)

            step += 1

            all_trajectories.append({
                "model": model_name,
                "episode": episode + 1,
                "step": step,
                "x_position": float(obs[0]),
                "y_position": float(obs[1]),
                "x_velocity": float(obs[2]),
                "y_velocity": float(obs[3]),
                "angle": float(obs[4]),
                "angular_velocity": float(obs[5]),
                "abs_angular_velocity": abs(float(obs[5])),
                "action": int(action)
            })

            done = terminated or truncated

    env.close()

df = pd.DataFrame(all_trajectories)

df.to_csv(
    "results/flight_trajectories.csv",
    index=False
)

print("Trajectory data saved!")
