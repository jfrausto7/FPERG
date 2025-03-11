import math
import numpy as np
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
from distributions.nominal_trajectory_dist import NominalTrajectoryDistribution
from distributions.proposal_trajectory_dist import ProposalTrajectoryDistribution
import pybullet as p
from typing import Dict, List

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
                except Exception as e:
                    print(f"Error: {e}")
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
        self.policy.reset()
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
                log_prob += np.log(dist.disturbance_distribution(t).pdf(elem) + epsilon)
        log_prob = np.clip(log_prob, -1e10, 1e10)
        return log_prob
    
    def stabilize_log_weights(self, log_weights):
        """
        Apply a smooth transformation to log weights to reduce extreme variations.
        """
        # Center around mean
        centered_log_weights = log_weights - np.mean(log_weights)
        # Apply a softer boundary using tanh
        scale_factor = 5.0
        stabilized_weights = np.tanh(centered_log_weights / scale_factor) * scale_factor
        return stabilized_weights

    def importanceSampling(self, d):
        # calculate number of samples based on n_trials
        m = max(20, min(60, int(20 + (np.log(self.n_trials) - np.log(10)) / (np.log(10000) - np.log(10)) * 40)))
        print(f"Running importance sampling with {m} samples...")

        # run multiple trials to get a more stable estimate
        num_trials = 3  # Using multiple trials for stability
        failure_probs = []
        std_errors = []
        
        for trial_idx in range(num_trials):
            # define nominal distribution
            pnom = NominalTrajectoryDistribution(d)
            # define proposal distribution: Tweak mean and std values to increase failure likelihood
            prop_dist = ProposalTrajectoryDistribution(0, 0.00015, d)
            
            # perform rollouts with proposal
            trajectories = [self.rollout(prop_dist, d) for _ in range(m)]
            
            # check if any trajectories resulted in failure
            failure_indicators = np.array([self.is_failure(trajectory) for trajectory in trajectories])
            failure_count = np.sum(failure_indicators)
            
            if failure_count == 0:
                print("Warning: No failures detected in trajectories.")
                failure_probs.append(0.0)
                std_errors.append(0.0)
                continue
                
            # compute log weights directly
            log_nom = np.array([self.logpdf(pnom, traj) for traj in trajectories])
            log_prop = np.array([self.logpdf(prop_dist, traj) for traj in trajectories])
            log_weights = log_nom - log_prop
            
            # apply stabilization to log weights (similar to adaptive IS)
            log_weights = self.stabilize_log_weights(log_weights)
            
            # stabilize and normalize weights
            max_log_weight = np.max(log_weights)
            stabilized_weights = np.exp(log_weights - max_log_weight)
            weight_sum = np.sum(stabilized_weights)
            
            if weight_sum < 1e-10:
                print("Warning: All weights are near zero. Using uniform weights.")
                normalized_weights = np.ones_like(stabilized_weights) / len(stabilized_weights)
            else:
                normalized_weights = stabilized_weights / weight_sum
            
            # compute weighted/unweighted failure probability
            weighted_failure_prob = np.sum(normalized_weights * failure_indicators)
            raw_failure_prob = failure_count / len(trajectories)
            
            # check if weights for failures are disproportionately low
            failure_weight_sum = np.sum(normalized_weights[failure_indicators])
            
            # Blend weighted and raw estimates for robustness
            diff = abs(failure_weight_sum / failure_count - raw_failure_prob)
            # more difference = rely more on simple average
            blend_alpha = min(0.5 + diff, 0.9)
            failure_probability = blend_alpha * raw_failure_prob + (1-blend_alpha) * weighted_failure_prob
            
            # compute variance and standard error
            if failure_probability > 0:
                variance = np.sum(normalized_weights**2 * (failure_indicators - failure_probability)**2)
                std_error = np.sqrt(variance)
            else:
                # if probability is 0, estimate standard error based on sample size
                std_error = np.sqrt((0 * (1-0)) / len(trajectories))
            
            failure_probs.append(failure_probability)
            std_errors.append(std_error)

        # final failure probability and standard error
        final_failure_prob = np.mean([p for p in failure_probs if p > 0]) if any(p > 0 for p in failure_probs) else 0.0
        final_std_error = np.sqrt(np.mean(np.array(std_errors)**2)) if any(std > 0 for std in std_errors) else 0.0

        return final_failure_prob, final_std_error