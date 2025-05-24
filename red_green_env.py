import gymnasium as gym
from gymnasium import spaces
import numpy as np

class RedGreenEnv(gym.Env):
    def __init__(self):
        super(RedGreenEnv, self).__init__()

        # Actions: 0 = stay still, 1 = move
        self.action_space = spaces.Discrete(2)

        # Observation: light state (0 = red, 1 = green)
        self.observation_space = spaces.Discrete(2)

        self.current_step = 0
        self.max_steps = 100  # episode length

        self.light_state = 1  # start with green

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.light_state = 1  # green light at start
        return self.light_state, {}

    def step(self, action):
        reward = 0

        if self.light_state == 1 and action == 1:
            reward = 1  # move on green light
        elif self.light_state == 0 and action == 1:
            reward = -10  # move on red light (penalty)
        else:
            reward = 0  # stay still no reward

        self.current_step += 1

        # Randomly switch light every step
        self.light_state = np.random.choice([0, 1])

        done = self.current_step >= self.max_steps
        truncated = False  # You can implement truncation if needed

        if done:
            reward += 50  # bonus for surviving full episode

        info = {}

        return self.light_state, reward, done, truncated, info

    def render(self, mode='human'):
        light = "Green" if self.light_state == 1 else "Red"
        print(f"Step: {self.current_step}, Light: {light}")
