import numpy as np
from environment import GraspEnv

class NoisySensor:
    def __init__(self, noise_std=0.01):
        """
        Initializes the sensor with a given noise standard deviation.
        :param noise_std: Standard deviation of Gaussian noise applied to sensor readings.
        """
        self.noise_std = noise_std

    def observe(self):
        """
        Returns a noisy observation of the true state.
        :param true_state: The actual state value (e.g., gripper position or force readings).
        :return: Noisy sensor measurement.
        """
        observation = GraspEnv.get_observation(self)
        gripper_true_state = observation[0:3]

        noise = np.random.normal(0, self.noise_std, size=gripper_true_state.shape)
        return gripper_true_state + noise