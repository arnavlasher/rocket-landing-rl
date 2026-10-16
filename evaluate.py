
import gymnasium as gym
from stable_baselines3 import PPO
import csv
import os

# Load the trained model
model = PPO.load("models/baseline_ppo")

# Create environment without graphics
env = gym.make("LunarLander-v3")

# Number of landing attempts
num_episodes = 100

# Store results
results = []

for episode in range(num_episodes):
    observation, info = env.reset(seed=1000 + episode)

    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Let the trained AI choose an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        # Execute action in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        done = terminated or truncated

    # Record whether the lander received a
    # positive landing termination reward.
    # This is a useful proxy, not a complete
    # landing-success classification.
    successful = (
        terminated
        and not truncated
        and reward >= 100
    )

    results.append({
        "episode": episode + 1,
        "reward": total_reward,
        "steps": steps,
        "successful_landing": int(successful)
    })

    print(
        f"Episode {episode + 1}/{num_episodes} | "
        f"Reward: {total_reward:.2f} | "
        f"Steps: {steps} | "
        f"Success: {successful}"
    )

env.close()

# Calculate performance metrics
average_reward = sum(
    r["reward"] for r in results
) / num_episodes

success_rate = sum(
    r["successful_landing"] for r in results
) / num_episodes * 100

average_steps = sum(
    r["steps"] for r in results
) / num_episodes

# Print summary
print("\n--- EVALUATION RESULTS ---")
print(f"Average Reward: {average_reward:.2f}")
print(f"Success Rate (proxy): {success_rate:.1f}%")
print(f"Average Steps: {average_steps:.1f}")

# Save results to CSV
os.makedirs("results", exist_ok=True)

with open("results/baseline_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "episode",
            "reward",
            "steps",
            "successful_landing"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nResults saved to results/baseline_results.csv")
