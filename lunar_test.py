import gymnasium as gym

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

# CHANGE: Using a 'while True' loop makes the simulation run forever
try:
    while True:
        action = env.action_space.sample()  # Take a random action
        observation, reward, terminated, truncated, info = env.step(action)
        
        # If the lander crashes or lands, reset it instantly and keep going
        if terminated or truncated:
            observation, info = env.reset()
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    env.close()
