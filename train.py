
import gymnasium as gym
from stable_baselines3 import PPO
import time

# Create the LunarLander environment
env = gym.make("LunarLander-v3")

# Create the PPO machine learning model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64
)

# Start training
start_time = time.time()

model.learn(total_timesteps=500_000)

# Calculate training time
end_time = time.time()
print(f"Training took {end_time - start_time:.2f} seconds")

# Save the trained model
model.save("rocket_ai")

env.close()

print("Training complete!")
