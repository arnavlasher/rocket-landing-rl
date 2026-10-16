
import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrappers import StabilityPrimeRewardWrapper
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Create LunarLander environment with stronger reward shaping
env = gym.make("LunarLander-v3")
env = StabilityPrimeRewardWrapper(env)

# Create PPO model with the same configuration
# used for the previous three models
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

# Save trained model
model.save("models/stability_prime_ppo")

env.close()

print("Model 3 Prime trained and saved successfully!")
