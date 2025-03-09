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
        Initialize importance sampling for failure probability estimation.

        Args:
            n_trials: Number of trials to run
            gui: Whether to use GUI mode
            use_hill_climbing: Whether to use hill climbing policy
            policy_file: Path to policy file for hill climbing
        """
        self.n_trials = n_trials
        self.gui = gui
        self.env = GraspEnv(gui=gui)

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


    """
    Rollout takes in the proposal distribution and a specific depth. 
    It generates trajectories based on the proposal distribution. 
    The initial random state is from the reset() function in GraspEnv
    The sensor disturbances x are sampled from the proposal distribution disturbance (sensor noise)
    These disturbances are added to each component of the state to get the observation
    We pass in the observation to the get_action() function to get the action
    Call the step function with the selected action to get the next state, reward, termination status, and success info
    We add the trajectory to the list and update the state s 
    """
    def rollout(self, proposal_distribution, depth):
        # Get random initial state
        s = self.env.reset()
        trajectory = []
        # Call step function for num_samples
        for t in range(depth):
            # This is where we sampled our disturbances and added them to the state to get a noisy observation
            o = s.copy()
            # Get disturbance distribution
            disturb_dist = proposal_distribution.disturbance_distribution(t)
            # For each elem of the state, sample from the disturbance distribution
            x = np.array([disturb_dist.rvs() for _ in range(len(s))])
            # Get the observation by adding disturbances to the state
            o += x
            # Action is based on observation received from sensor
            a = self.policy.get_action(o)
            # s_prime is get_observation() returns a np array of gripper and object position
            s_prime, reward, done, info = self.env.step(a)
            # Trajectory is a list of dicts
            trajectory.append({'state': s, 'obs': o, 'action': a, 'disturbance': x, 'done': done, 'success': info['grasp_success']})
            s = s_prime

            if done:
                break
        return trajectory

    """
    Check if a trajectory led to a failure
    - Check if the final step in the trajectory is a success or failure
    """
    def is_failure(self, trajectory):
        # Failure only if final step fails
        return not trajectory[-1]['success']

    """
    Logpdf function we get the likelihood of the samples we already have with respect to 
    our initial state distributions and disturbance distributions for the 
    nominal or proposal distribution
    
    Get the sample of x, y values from our randomly initial state
    - Take the pdf of sampling THOSE x, y values from our random uniforms 
    
    Same with the disturbances:
    For the disturbances added to sensor 
    - What is the likelihood of sampling each value from our disturbance distribution
    """
    def logpdf(self, dist, trajectory):
        log_prob = 0
        # Get likelihood of first state of trajectory
        x_obj_pos, y_obj_pos = trajectory[0]['state'][3], trajectory[0]['state'][4]
        #print(f"x_obj_pos: {x_obj_pos}")
        #print(f"y_obj_pos: {y_obj_pos}")

        # keep the positions in bounds
        epsilon = 1e-8
        # x_obj_pos = np.sign(x_obj_pos)*(max(abs(x_obj_pos) - 0.0001, 0))
        # y_obj_pos = np.sign(y_obj_pos)*(max(abs(y_obj_pos) - 0.0001, 0))

        log_prob += np.log(dist.initial_state_distribution()[0].pdf(x_obj_pos) + epsilon)
        log_prob += np.log(dist.initial_state_distribution()[1].pdf(y_obj_pos) + epsilon)

        # Go through each prob in disturbance and add it to the log prob
        for t in range(len(trajectory)):
            for elem in trajectory[t]['disturbance']:
                log_prob += np.log(dist.disturbance_distribution(t).pdf(elem))
        log_prob = np.clip(log_prob, -1e10, 1e10)
        return log_prob

    def importanceSampling(self, d):
        # Calculate number of samples
        #print(f"number trials: {self.n_trials}")
        #print(f"depth: {d}")
        m = self.n_trials // d
        #print(f"number of samples: {m}")


        # Define nominal distribution
        pnom = NominalTrajectoryDistribution(d)
        # Define proposal distribution: Tweak mean and std values to increase failure likelihood
        prop_dist = ProposalTrajectoryDistribution(0, 0.75, d)
        # Perform rollouts with proposal
        trajectories = [self.rollout(prop_dist, d) for _ in range(m)]

        # Compute log weights directly
        log_nom = np.array([self.logpdf(pnom, traj) for traj in trajectories])
        log_prop = np.array([self.logpdf(prop_dist, traj) for traj in trajectories])
        log_weights = log_nom - log_prop
        stabilized_weights = np.exp(log_weights - np.max(log_weights))  # Stabilize numerically
        normalized_weights = stabilized_weights / np.sum(stabilized_weights)

        # Compute weighted average of samples from the proposal distribution
        weighted_samples = [w * self.is_failure(trajectory) for w, trajectory in zip(normalized_weights, trajectories)]
        failure_probability = np.mean(weighted_samples)
        n = len(weighted_samples)
        variance = np.sum((weighted_samples - failure_probability)**2) / (n)
        std_error = np.sqrt(variance / n)   # this should go down over time...

        return failure_probability, std_error

