
import gymnasium as gym
from stable_baselines3 import PPO

# Create the simulation with graphics enabled
env = gym.make("LunarLander-v3", render_mode="human")

# Load the trained neural network
model = PPO.load("models/baseline_ppo")

# Watch the AI attempt 10 landings
for episode in range(10):
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # AI chooses an action using its neural network
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1}: Reward = {total_reward:.2f}")

env.close()
