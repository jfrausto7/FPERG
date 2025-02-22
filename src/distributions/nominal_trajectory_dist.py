import numpy as np
from scipy.stats import norm, uniform
import random
from environment.GraspEnv import GraspEnv

class NominalTrajectoryDistribution:
    def __init__(self, d):
        # Should define nominal values for mean and std for the sensor noise (try standard normal)
        self.mean = 0
        self.std = 1
        self.state_len = 6
        self.depth = d


    # return two values from a random uniform(-1.1)
    # I can reverse enginner the x, y components of my grasp env to return two distributions
    # 0.5 + 0.05 * random.uniform(-1, 1) -> (-0.5/uniform(-1, 1)) - 0.05 (I think?)

    # Initial state distribution includes object and gripper position
    def initial_state_distribution(self):
        # NEED TO DOUBLE CHECK THIS ADJUSTMENT OF THE UNIFORM
        return [uniform(0.45, 0.55), uniform(-0.05, 0.05)]

    # This should be an object you can sample from NOT a sample itself

    # This will be the get_observation function in grasp_env
    # Add some gaussian noise to the true observation
    """ We add a disturbance only on the system's sensor. 
    The agent and environment are deterministic.
    Sensor is the observation of our environment and we add gaussian noise to 
    imitate imperfect sensor readings in the real world. """

    def disturbance_distribution(self, t):
        # Numpy array of six values of gripper and object [x_g, y_g, z_g, x_o, y_o, z_o]
        # sensor_disturbance = GraspEnv.get_observation()
        # Either return a list of 6 gaussians or Sample from the disturbance dist 6 times
        return norm(self.mean, self.std)
        #return [norm(self.mean, self.std) for _ in range(self.state_len)]

    def depth(self):
        return self.depth
