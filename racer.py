from gym.spaces import MultiBinary, Box
import numpy as np
import cv2
import gym


class RacingGym(gym.Env):

    """Modified Gym wrapper for racing game to add preprocessing steps to observations"""

    def __init__(self):
        """Gym initialiser"""
        super().__init__()
        # Specify action space and observation space
        self.observation_space = Box(low=0, high=255, shape=(96, 96, 1), dtype=np.uint8)
        self.action_space = Box(np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
        self.game = gym.make("CarRacing-v0")

    def reset(self):
        """Reset the state of the game"""
        # Return the first frame
        obs = self.game.reset()
        obs = self.preprocess(obs)
        return obs

    def preprocess(self, observation: Box):
        """preporcess an image to gray scale and de-noise"""
        obs_grey = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        obs_standardised = (obs_grey - np.min(obs_grey)) / (np.max(obs_grey) - np.min(obs_grey))
        return np.resize(obs_standardised, (96, 96, 1))

    def step(self, action: MultiBinary):
        """take a step in the game"""
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        """render game frame"""
        self.game.render()

    def close(self):
        """close game instance"""
        self.game.close()


if __name__ == "__main__":
    pass
