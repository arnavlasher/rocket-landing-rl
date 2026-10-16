
import gymnasium as gym

class TimePenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.01):
        super().__init__(env)
        self.penalty = penalty

    def step(self, action):
        # Take an action in the original environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply our custom time penalty
        reward -= self.penalty

        # Return the modified reward
        return observation, reward, terminated, truncated, info
