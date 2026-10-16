
import gymnasium as gym

# Create the simulation
env = gym.make("LunarLander-v3", render_mode="human")

# Names of the 8 observation values
names = [
    "Horizontal Position",
    "Vertical Position",
    "Horizontal Velocity",
    "Vertical Velocity",
    "Angle",
    "Angular Velocity",
    "Left Leg Contact",
    "Right Leg Contact"
]

# Run 5 random landing attempts
for episode in range(5):
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose a random action
        action = env.action_space.sample()

        # Perform the action
        observation, reward, terminated, truncated, info = env.step(action)

        # Print the 8 observations
        for i in range(8):
            print(f"{names[i]}: {observation[i]:.3f}")

        print(f"Action: {action}")
        print(f"Reward: {reward:.3f}")
        print("--------------------")

        total_reward += reward
        done = terminated or truncated

    print(f"Episode {episode + 1} finished!")
    print(f"Total Reward: {total_reward:.2f}")

env.close()
