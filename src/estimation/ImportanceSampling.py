import numpy as np
from scipy.stats import norm, uniform
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
from distributions.nominal_trajectory_dist import NominalTrajectoryDistribution
from distributions.proposal_trajectory_dist import ProposalTrajectoryDistribution
import pandas as pd
import pybullet as p
from typing import Dict, List, Tuple
import time

class importanceSamplingEstimation:
    def __init__(self, n_trials: int = 1000, gui: bool = False, use_hill_climbing: bool = False,
                 policy_file: str = None):
        """
        Initialize direct estimation for failure probability estimation.

        Args:
            n_trials: Number of trials to run
            gui: Whether to use GUI mode
            use_hill_climbing: Whether to use hill climbing policy
            policy_file: Path to policy file for hill climbing
        """
        self.n_trials = n_trials
        self.gui = gui
        self.env = GraspEnv(gui=gui)
        self.nominal_distribution = None
        self.proposal_distribution = None

        # initialize appropriate policy
        if use_hill_climbing:
            self.policy = HillClimbingGraspingPolicy()
            if policy_file:
                try:
                    self.policy.load_policy(policy_file)
                    print(f"Loaded hill climbing policy from {policy_file}")
                except:
                    print(f"Warning: Could not load policy from {policy_file}")
            self.policy.epsilon = 0  # disable exploration for evaluation
        else:
            self.policy = GraspingPolicy()

        self.results: List[Dict] = []

    def rollout(self, proposal_distribution, depth):
        # Get random initial state
        s = self.env.reset()
        #print(s)
        #print(f"Shape of s: {s.shape}")  # Debugging: check the shape of o
        trajectory = []
        # Call step function for num_samples
        for t in range(depth):
            # This is where we sampled our disturbances and added them to the state to get a noisy observation
            x = []
            o = s.copy()
            # Sample disturbances for each elem in s from the disturbance distribution
            for i in range(len(s)):
                disturbance = proposal_distribution.disturbance_distribution(t) # We don't need to iterate over `i`
                #print(f"disturbance at {i}: {disturbance}")
                x.append(disturbance.rvs())
            x = np.array(x)
            #print(x)
            #print(f"Shape of x: {x.shape}")  # Debugging: check the shape of o
            for i in range(len(x)):
                o[i] = s[i] + x[i]
            #print(o)
            #print(f"Shape of obs: {o.shape}")  # Debugging: check the shape of o

            a = self.policy.get_action(o) # Action is based on observation received from sensor
            # s_prime is get_observation() returns a np array of gripper and object position
            s_prime, reward, done, info = self.env.step(a)

            # Trajectory is a list of dicts
            trajectory.append({'state': s, 'obs': o, 'action': a, 'disturbance': x, 'success': info['grasp_success']})
            s = s_prime

            if done:
                break
        print(f"Returned trajectory: {trajectory}")
        return trajectory

    """
    Check if a trajectory led to a failure
    """
    """def is_failure(self, trajectory):
        # If the trajectory was a success return false (it wasn't a failure)
        if trajectory['success']:
            return False
        else:
            return True"""

    def is_failure(self, trajectory):
        # Check if any depth in the trajectory is a failure
        for step in trajectory:
            if not step['success']:  # Assuming 'success' is False for failure
                return True  # If any step fails, consider the trajectory a failure
        return False  # If no steps fail, the trajectory is not a failure

    """
    In the pdf function, WE DON'T WANT TO TAKE SAMPLES WE ARE ALREADY HAVE SAMPLES. 
    What we want to do is get the likelihood of the samples we already have with respect to 
    our initial state distributions and disturbance distributions 
    
    Get the sample of x, y values from our randomly initial state
    - Take the pdf of sampling THOSE x, y values from our random uniforms 
    pdf of x = uniform.pdf(x, loc=0.45, scale=0.1)
    pdf of y = uniform.pdf(y, loc=-0.05, scale=0.1)
    
    Same with the disturbances:
    For the disturbances added to sensor 
    - What is the likelihood of sampling each value from our disturbance distribution
    for elem in disturbance values:
        pdf += norm.pdf(elem, loc=0, scale=1) Or something like this 
    """

    def logpdf(self, dist, trajectory):
        log_prob = 0
        # Get likelihood of first state of trajectory
        x_obj_pos = trajectory[0]['state'][3]
        print(f"x_obj_pos: {x_obj_pos}")
        y_obj_pos = trajectory[0]['state'][4]
        print(f"y_obj_pos: {y_obj_pos}")

        #dist.initial_state_distribution[0] -> this should return a uniform(0.45, 0.55)
        # Then I went to get the pdf of the x_obj_pos from this distribution
        #log_prob += dist.initial_state_distribution()[0].pdf(x_obj_pos)
        #prob_x = np.log(dist.initial_state_distribution()[0].pdf(x_obj_pos))
        log_prob += np.log(dist.initial_state_distribution()[0].pdf(x_obj_pos))

        # dist.initial_state_distribution[0] -> this should return a uniform(-0.05, 0.05)
        # Then I went to get the pdf of the y_obj_pos from this distribution
        #prob_y = dist.initial_state_distribution()[1].pdf(y_obj_pos)
        #log_prob_y = np.log(dist.initial_state_distribution()[1].pdf(y_obj_pos))
        log_prob += np.log(dist.initial_state_distribution()[1].pdf(y_obj_pos))

        print(f"log_prob: {log_prob}")
        # Go through each prob in disturbance and add it to the log prob
        """        
        Need to be able to pass a time variable into the disturbance distribution.
        In the rollout we looped through the depth which gave us t
        What is t in this case of finding the logpdf?
        """
        for t in range(len(trajectory)):
            for elem in trajectory[t]['disturbance']:
            # get prob of drawing sample from the disturbance distribution
            # dist.disturbance_distribution() -> this should return a norm(mean, std)
            # I want to get the pdf of sampling elem from this distribution
                log_prob += np.log(dist.disturbance_distribution(t).pdf(elem))
        return log_prob

    def importanceSampling(self, d):
        # Calculate number of samples
        #print(f"number trials: {self.n_trials}")
        #print(f"depth: {d}")
        m = self.n_trials // d
        #print(f"number of samples: {m}")


        # Define nominal distribution
        pnom = NominalTrajectoryDistribution(d)
        # Define proposal distribution: Tweak mean and covariance values to increase failure likelihood
        prop_dist = ProposalTrajectoryDistribution(0, 1, d)
        # Perform rollouts with proposal
        trajectories = [self.rollout(prop_dist, d) for _ in range(m)]

        # Calculate likelihoods
        nom_likelihood = np.array([np.exp(self.logpdf(pnom, trajectory)) for trajectory in trajectories])
        prop_likelihood = np.array([np.exp(self.logpdf(prop_dist, trajectory)) for trajectory in trajectories])

        # turn likelihoods into nparrays to do element wise division
        weights = nom_likelihood / prop_likelihood
        failure_probability = np.mean([w * self.is_failure(trajectory) for w, trajectory in zip(weights, trajectories)])

        return failure_probability

