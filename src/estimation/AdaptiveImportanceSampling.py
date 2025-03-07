import numpy as np
import math
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
from distributions.nominal_trajectory_dist import NominalTrajectoryDistribution
from distributions.proposal_trajectory_dist import ProposalTrajectoryDistribution
from typing import Dict, List

class adaptiveImportanceSamplingEstimation:
    def __init__(self, n_trials: int = 1000, gui: bool = False, use_hill_climbing: bool = False,
                 policy_file: str = None):
        """
        Initialize adaptive importance sampling for failure probability estimation.

        Args:
            n_trials: Number of trials to run
            gui: Whether to use GUI mode
            use_hill_climbing: Whether to use hill climbing policy
            policy_file: Path to policy file for hill climbing
        """
        self.n_trials = n_trials
        self.gui = gui
        self.env = GraspEnv(gui=gui)
        self.required_lift_duration = 3.0  # duration for successful lift


        # initialize appropriate policy
        if use_hill_climbing:
            self.policy = HillClimbingGraspingPolicy()
            if policy_file:
                try:
                    self.policy.load_policy(policy_file)
                    print(f"Loaded hill climbing policy from {policy_file}")
                except Exception as e:
                    print (f"Error: {e}")
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
            # get object position from state
            obj_pos = s[3:6]
            # Trajectory is a list of dicts
            trajectory.append({
                        'state': s, 
                        'obs': o, 
                        'action': a, 
                        'disturbance': x, 
                        'done': done, 
                        'success': info['grasp_success'],
                        'lifted': obj_pos[2] - self.env.initial_obj_height > self.env.lift_threshold,
                        'lift_duration': info.get('lift_duration', 0)
                    })
            s = s_prime

            if done:
                break
        # print(f"Returned trajectory: {trajectory}")
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

        log_prob += np.log(dist.initial_state_distribution()[0].pdf(x_obj_pos))
        log_prob += np.log(dist.initial_state_distribution()[1].pdf(y_obj_pos))

        # Go through each prob in disturbance and add it to the log prob
        for t in range(len(trajectory)):
            for elem in trajectory[t]['disturbance']:
                log_prob += np.log(dist.disturbance_distribution(t).pdf(elem))
                
        log_prob = np.clip(log_prob, -1e10, 1e10)
        # print(f"log_prob: {log_prob}")
        return log_prob

    """
    Objective function that will return values greater than 0 for successes and
    less than or equal to 0 for failures.

    Computes robustness for a trajectory: >0 for success, ≤0 for failures.
    Flow:
    1. Success: If not a failure (final step success), return positive robustness based on lift duration excess + 1.
    2. Failure Case 1: If object was lifted but trajectory failed, return negative robustness as max lift duration minus required duration.
    3. Failure Case 2: If never lifted, return negative robustness as the minimum Euclidean distance between gripper and object across steps.
    """
    def simple_failure_function(self, trajectory): 
        # For a last state of a trajectory, check if it’s a success -> positive value 
        if not self.is_failure(trajectory):
            robustness = max(trajectory[-1]['lift_duration'] - self.required_lift_duration, 0) + 1  # Base success = 1
            return robustness

        # Check if the object was ever lifted
        lifted = any([step['lifted'] for step in trajectory])
        
        # Failure CASE 1: Lifted object but dropped it, so ended in failure 
        if lifted:
            max_lift_duration = max([step['lift_duration'] for step in trajectory])
            robustness = max_lift_duration - self.required_lift_duration 	
            return robustness
        
        # Failure CASE 2: Missed the object completely 	
        else:
            min_distance = float('inf')
            
            for step in trajectory:
                gripper_pos = np.array(step['state'][:3])
                obj_pos = np.array(step['state'][3:])

                # Euclidean distance between gripper and object
                distance = np.linalg.norm(gripper_pos - obj_pos)
                min_distance = min(min_distance, distance)
            
            robustness = -min_distance 
            return robustness
        
    """
    Fit a new proposal distribution based on weighted trajectories.
    
    Args:
        q: Current proposal distribution
        trajectories: List of trajectory samples
        ws: Sample weights
    
    Returns:
        A new proposal distribution with updated parameters
    """
    def fit(self, q, trajectories, ws):

        # normalize weights (in case they aren't already)
        normalized_ws = np.array(ws) / np.sum(ws) if np.sum(ws) > 0 else np.ones_like(ws) / len(ws)
        
        # extract all disturbances from trajs weighted by their importance
        disturbance_samples = []
        for trajectory, weight in zip(trajectories, normalized_ws):
            for step in trajectory:
                disturbance_samples.extend([(d, weight) for d in step['disturbance']])
    
        disturbances, weights = zip(*disturbance_samples) if disturbance_samples else ([], [])
        disturbances = np.array(disturbances)
        weights = np.array(weights)

        if len(disturbances) > 0:
            weighted_mean = np.sum(disturbances * weights.reshape(-1, 1), axis=0) / np.sum(weights)
            weighted_var = np.sum(weights.reshape(-1, 1) * (disturbances - weighted_mean)**2, axis=0) / np.sum(weights)
            weighted_std = np.sqrt(weighted_var)
            
            # ensure minimum std to prevent degeneration and create new proposal
            avg_std = np.mean(weighted_std)
            min_std = 0.1
            avg_std = max(min_std, avg_std)
            # weighted_std = max(0.1, weighted_std)            
            new_q = ProposalTrajectoryDistribution(float(weighted_mean[0]), float(avg_std), q.depth)
            return new_q
        
        # slightly widen the current distribution when there are no valid disturbances
        return ProposalTrajectoryDistribution(q.mean, q.std * 1.1, q.depth)

    """
    Iterates to find a proposal distribution for importance sampling.
    Note: this will do nothing until the fit function is correctly implemented
    """

    def find_proposal_dist(self, nominal_dist, proposal_dist, f, k_max, m, m_elite, d):
        for k in range(k_max):
            # Perform rollouts with proposal
            trajectories = [self.rollout(proposal_dist, d) for _ in range(m)]
            Y = [f(traj) for traj in trajectories]
            Y.sort()
            cutoff = max(0, Y[m_elite])
            ps = np.array([self.logpdf(nominal_dist, traj) for traj in trajectories])
            qs = np.array([self.logpdf(proposal_dist, traj) for traj in trajectories])
            ws = ps / qs
            ws = [w if y < cutoff else 0 for w,y in zip(ws,Y)]
            proposal_dist = self.fit(proposal_dist, trajectories, ws)
        return proposal_dist

    """
    Implements adaptive importance sampling to return a failure probability estimation.
    """

    def adaptiveImportanceSampling(self, d):
        # Calculate number of samples
        m = self.n_trials // d
        m_elite = math.ceil(m / 10) - 1
        k_max = self.n_trials
        f = self.simple_failure_function

        # Define nominal distribution
        pnom = NominalTrajectoryDistribution(d)
        # Define initial proposal distribution
        init_prop_dist = ProposalTrajectoryDistribution(0, 0.75, d)
        # Define proposal distribution: Tweak mean and std values to increase failure likelihood
        prop_dist = self.find_proposal_dist(pnom, init_prop_dist, f, k_max, m, m_elite, d)

        # Perform rollouts with final proposal
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
        std_error = np.std(weighted_samples)

        return failure_probability, std_error

