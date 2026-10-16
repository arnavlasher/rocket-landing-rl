
import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrappers import StabilityRewardWrapper
import os
import time

# Create environment
env = gym.make("LunarLander-v3")

# Apply stability reward shaping
env = StabilityRewardWrapper(
    env,
    time_penalty=0.01,
    stability_penalty=0.02
)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# Train model
start_time = time.time()

model.learn(total_timesteps=500_000)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.2f} seconds")

# Save model
os.makedirs("models", exist_ok=True)

model.save("models/stability_ppo")

env.close()

print("Stability model saved successfully!")
