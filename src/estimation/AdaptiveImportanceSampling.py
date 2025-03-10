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
                            
        return trajectory

    """
    Check if a trajectory led to a failure
    - Check if the final step in the trajectory is a success or failure
    """
    def is_failure(self, trajectory):
        # Failure only if final step fails
        result = not trajectory[-1]['success']
        return result

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
        # For a last state of a trajectory, check if it's a success -> positive value 
        is_fail = self.is_failure(trajectory)
        
        if not is_fail:
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
            
            # lambda controls how close we stay to nominal (higher = closer to nominal)
            lambda_reg = 0.95
            regularized_mean = (1 - lambda_reg) * weighted_mean[0]  # pull strongly toward nominal mean (0)
            
            # limit how far the mean can move from nominal in each iteration
            max_step = 0.0001  # max allowed change in mean per iteration
            if abs(regularized_mean) > max_step:
                regularized_mean = np.sign(regularized_mean) * max_step
            
            # ensure minimum std to prevent degeneration and create new proposal
            avg_std = np.mean(weighted_std)
            min_std = 0.0001
            max_std = 0.0003  # min/max std to prevent too wide exploration
            avg_std = max(min_std, min(max_std, avg_std))  # clamp between min and max
            
            # Create new proposal with regularized parameters
            new_q = ProposalTrajectoryDistribution(float(regularized_mean), float(avg_std), q.depth)
            
            return new_q
        
        # slightly widen the current distribution when there are no valid disturbances
        return ProposalTrajectoryDistribution(q.mean, q.std * 1.0001, q.depth)

    """
    Iterates to find a proposal distribution for importance sampling.
    Note: this will do nothing until the fit function is correctly implemented
    """

    def find_proposal_dist(self, nominal_dist, proposal_dist, f, k_max, m, m_elite, d):
        for k in range(k_max):
            if k % 10 == 0:
                print(f"Completed {k}/{k_max} iterations...")
            # Perform rollouts with proposal
            trajectories = [self.rollout(proposal_dist, d) for _ in range(m)]
            
            # Check if trajectories list is empty
            if not trajectories:
                return proposal_dist
                
            Y = [f(traj) for traj in trajectories]
            Y.sort()
            
            # Safe access to Y with index checking
            if m_elite < len(Y):
                cutoff = max(0, Y[m_elite])
            else:
                cutoff = max(0, Y[-1]) if Y else 0
                
            ps = np.array([self.logpdf(nominal_dist, traj) for traj in trajectories])
            qs = np.array([self.logpdf(proposal_dist, traj) for traj in trajectories])
            ws = ps / qs
            ws = [w if y < cutoff else 0 for w,y in zip(ws,Y)]
            proposal_dist = self.fit(proposal_dist, trajectories, ws)
        return proposal_dist

    """
    Implements adaptive importance sampling to return a failure probability estimation.
    """

    def adaptiveImportanceSampling(self, d, k_max=10):
        # Calculate number of samples
        m = max(5, int(5 + (np.log(self.n_trials) - np.log(250)) / (np.log(10000) - np.log(250)) * 5))  # range: 5-10
        m_elite = max(1, int(np.ceil(m / 3) - 1))  # range: 1-3

        # k_max = self.n_trials
        f = self.simple_failure_function

        # Define nominal distribution
        pnom = NominalTrajectoryDistribution(d)
        # Define initial proposal distribution
        init_prop_dist = ProposalTrajectoryDistribution(0, 0.000125, d)
        
        # Define proposal distribution: Tweak mean and std values to increase failure likelihood
        prop_dist = self.find_proposal_dist(pnom, init_prop_dist, f, k_max, m, m_elite, d)

        # sample half from nominal, half from proposal so it doesn't stray too far away:
        nom_trajectories = [self.rollout(pnom, d) for _ in range(m//2)]
        prop_trajectories = [self.rollout(prop_dist, d) for _ in range(m//2)]
        trajectories = nom_trajectories + prop_trajectories

        # Compute log weights directly
        log_nom = np.array([self.logpdf(pnom, traj) for traj in trajectories])
        log_prop = np.array([self.logpdf(prop_dist, traj) for traj in trajectories])
        log_weights = log_nom - log_prop
        
        # Before doing max stabilization, check for extreme values
        if np.max(log_weights) - np.min(log_weights) > 20:  # handle when weights are extremely different
            
            # Option 1: Cap the difference to avoid one weight dominating
            mean_log_weight = np.mean(log_weights)
            max_diff = 5.0  # Maximum allowed difference from mean (adjust as needed)
            log_weights = np.clip(log_weights, mean_log_weight - max_diff, mean_log_weight + max_diff)
        
        # Then proceed with stabilization and normalization
        max_log_weight = np.max(log_weights)
        stabilized_weights = np.exp(log_weights - max_log_weight)
        
        # Ensure weights are properly normalized
        if np.sum(stabilized_weights) > 1e-10:
            normalized_weights = stabilized_weights / np.sum(stabilized_weights)
        else:
            # Fall back to uniform weights if numerical issues occur
            normalized_weights = np.ones_like(stabilized_weights) / len(stabilized_weights)
                
        # Compute weighted average of samples from the proposal distribution
        failure_indicators = np.array([self.is_failure(trajectory) for trajectory in trajectories])
        
        # Add additional sanity check before final calculation
        if np.sum(failure_indicators) > 0:
            # Check if weights for failures are disproportionately high
            failure_weight_sum = np.sum(normalized_weights[failure_indicators])
            failure_count = np.sum(failure_indicators)
            failure_ratio = failure_count / len(failure_indicators)
                        
            # If weight sum for failures is vastly different from actual proportion
            if failure_weight_sum > 0.9 and failure_ratio < 0.5:
                
                # Option 2: Use a blend of weighted and simple averages
                simple_failure_prob = failure_ratio
                alpha = 0.7  # blend factor (may need to adjust)
                blended_prob = alpha * simple_failure_prob + (1-alpha) * failure_weight_sum
                failure_probability = blended_prob
            else:
                # Normal weighted calculation
                failure_probability = np.sum(normalized_weights * failure_indicators)
        else:
            failure_probability = 0.0
        
        # Calculate variance and standard error
        if failure_probability > 0:
            variance = np.sum(normalized_weights**2 * (failure_indicators - failure_probability)**2)
            std_error = np.sqrt(variance)
        else:
            # If probability is 0, estimate standard error based on sample size
            std_error = np.sqrt((0 * (1-0)) / len(trajectories))
        
        return failure_probability, std_error