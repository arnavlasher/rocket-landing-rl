
import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrappers import StabilityDoublePrimeRewardWrapper
import os

# Create models directory
os.makedirs("models", exist_ok=True)

# Create LunarLander environment
env = gym.make("LunarLander-v3")

# Apply intermediate stability reward shaping
env = StabilityDoublePrimeRewardWrapper(env)

# Create PPO model with same hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# Train for 500,000 timesteps
model.learn(total_timesteps=500_000)

# Save model
model.save("models/stability_double_prime_ppo")

env.close()

print("Model 3 Double Prime trained and saved successfully!")
