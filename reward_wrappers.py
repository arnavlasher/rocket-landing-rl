
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


class StabilityRewardWrapper(gym.Wrapper):
    def __init__(self, env, time_penalty=0.01, stability_penalty=0.02):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.stability_penalty = stability_penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Angular velocity is observation index 5
        angular_velocity = observation[5]

        # Penalize taking too long
        reward -= self.time_penalty

        # Penalize excessive rotation
        reward -= self.stability_penalty * abs(angular_velocity)

        return observation, reward, terminated, truncated, info

class StabilityPrimeRewardWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        reward -= 0.01
        reward -= 0.20 * abs(observation[5])

        return observation, reward, terminated, truncated, info


    
class StabilityDoublePrimeRewardWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Penalize taking too long
        reward -= 0.01

        # Intermediate angular velocity penalty
        reward -= 0.05 * abs(observation[5])

        return observation, reward, terminated, truncated, info
