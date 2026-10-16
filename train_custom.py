
import gymnasium as gym
from stable_baselines3 import PPO
from reward_wrappers import TimePenaltyWrapper
import time

# Create the LunarLander environment
env = gym.make("LunarLander-v3")

# Apply our custom reward function
env = TimePenaltyWrapper(env, penalty=0.01)

# Create a new PPO model
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

# Record training time
training_time = time.time() - start_time

print(f"Training took {training_time:.2f} seconds")

# Save the new model
model.save("models/time_penalty_ppo")

env.close()

print("Custom reward training complete!")
