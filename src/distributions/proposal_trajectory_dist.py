import numpy as np
from scipy.stats import norm, uniform
import random
from environment.GraspEnv import GraspEnv

class ProposalTrajectoryDistribution:
    def __init__(self, mean, std, d):
        # Choose values of mean and std that will make it more likely for system to fail
        self.mean = mean
        self.std = std
        self.state_len = 6
        self.depth = d

    # Initial state distribution includes object and gripper position
    def initial_state_distribution(self):
        # I could add disturbance to the initial state to make failures more likely
        return [uniform(0.45, 0.55), uniform(-0.05, 0.05)]

    # This will be the get_observation function in grasp_env
    # Add some gaussian noise to the true observation
    """ We add a disturbance only on the system's sensor. 
    The agent and environment are deterministic.
    Sensor is the observation of our environment and we add EXTRA gaussian noise to 
    stress the system and produce failures. """
    def disturbance_distribution(self, t):
        # Numpy array of six values of gripper and object [x_g, y_g, z_g, x_o, y_o, z_o]
        #sensor_disturbance = GraspEnv.get_observation()
        return norm(self.mean, self.std)
        #return [norm(self.mean, self.std) for _ in range(self.state_len)]

    def depth(self):
        return self.depth
