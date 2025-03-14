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
        self.required_lift_duration = 3.0  # duration (seconds) for successful lift

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
        x = np.zeros(len(s))
        # Call step function for num_samples
        for t in range(depth):
            # This is where we sampled our disturbances and added them to the state to get a noisy observation
            o = s.copy()
            # if depth is greater than 500 apply noisy disturbances
            if t >= 500:
                # Get disturbance distribution
                disturb_dist = proposal_distribution.disturbance_distribution(t)
                #    For each elem of the state, sample from the disturbance distribution
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
        for t in range(500, len(trajectory)):
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
        weights_samples = []
        for trajectory, weight in zip(trajectories, normalized_ws):
            for step in trajectory:
                # handle both scalar and array disturbances
                if np.isscalar(step['disturbance']) or len(step['disturbance']) == 0:
                    continue  # Skip empty or scalar disturbances
                disturbance_samples.extend(step['disturbance'])
                weights_samples.extend([weight] * len(step['disturbance']))
        
        # check if if we have any valid disturbances (500+ depth)
        if not disturbance_samples:
            min_std = 0.0001
            return ProposalTrajectoryDistribution(q.mean, max(q.std * 1.05, min_std), q.depth)
        
        disturbances = np.array(disturbance_samples)
        weights = np.array(weights_samples)

        weighted_mean = 0.0 
        for d, w in zip(disturbances, weights):
            weighted_mean += d * w
        weighted_mean /= np.sum(weights)
        
        weighted_var = 0.0
        for d, w in zip(disturbances, weights):
            weighted_var += w * (d - weighted_mean)**2
        weighted_var /= np.sum(weights)
        weighted_std = np.sqrt(weighted_var)
        
        # lambda controls how close we stay to nominal (higher = closer to nominal)
        lambda_reg = 0.95
        # use weighted_mean directly if it's a scalar
        regularized_mean = (1 - lambda_reg) * weighted_mean
        
        # limit how far the mean can move from nominal in each iteration
        max_step = 0.0001  # max allowed change in mean per iteration
        if abs(regularized_mean) > max_step:
            regularized_mean = np.sign(regularized_mean) * max_step
        
        # ensure minimum std to prevent degeneration and create new proposal
        avg_std = weighted_std if np.isscalar(weighted_std) else np.mean(weighted_std)
        min_std = 0.0001
        max_std = 0.001  # min/max std to prevent too wide exploration
        
        # apply smoothing to std changes
        if hasattr(self, 'prev_std'):
            smoothing_factor = 0.7
            avg_std = smoothing_factor * self.prev_std + (1 - smoothing_factor) * avg_std
            
        avg_std = max(min_std, min(max_std, avg_std))  # clamp between min and max
        self.prev_std = avg_std  # Store for next iteration
        
        # new proposal with regularized parameters
        new_q = ProposalTrajectoryDistribution(float(regularized_mean), float(avg_std), q.depth)
        
        return new_q

    def find_proposal_dist(self, nominal_dist, proposal_dist, f, k_max, m, m_elite, d):
        self.prev_std = proposal_dist.std
        
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
                
            # stable weight calculation
            log_ps = np.array([self.logpdf(nominal_dist, traj) for traj in trajectories])
            log_qs = np.array([self.logpdf(proposal_dist, traj) for traj in trajectories])
            
            # compute log ratio and stabilize
            log_ws = log_ps - log_qs
            max_log_w = np.max(log_ws)
            ws = np.exp(log_ws - max_log_w)
            
            # cutoff based on f value
            ws = [w if y < cutoff else 0 for w, y in zip(ws, Y)]
            
            proposal_dist = self.fit(proposal_dist, trajectories, ws)
            
        return proposal_dist

    def stabilize_log_weights(self, log_weights):
        """
        Apply a smooth transformation to log weights to reduce extreme variations.
        """
        # center around mean
        centered_log_weights = log_weights - np.mean(log_weights)
        # apply a softer boundary using tanh
        scale_factor = 5.0
        stabilized_weights = np.tanh(centered_log_weights / scale_factor) * scale_factor
        return stabilized_weights

    def adaptiveImportanceSampling(self, d, k_max=20):
        # Calculate number of samples
        m = self.n_trials
        m_elite = max(1, m // 10)
        print("Number of samples:", m)
        print("Number of elite samples:", m_elite)

        f = self.simple_failure_function

        # Define nominal distribution
        pnom = NominalTrajectoryDistribution(d)
        # Define initial proposal distribution
        init_prop_dist = ProposalTrajectoryDistribution(0, 0.0001025, d)
        
        # Define proposal distribution: Tweak mean and std values to increase failure likelihood
        prop_dist = self.find_proposal_dist(pnom, init_prop_dist, f, k_max, m, m_elite, d)

        # Run multiple trials to get a more stable estimate
        num_final_trials = 2
        failure_probs = []
        std_errors = []
        
        for trial_idx in range(num_final_trials):
            # sample half from nominal, half from proposal:
            nom_trajectories = [self.rollout(pnom, d) for _ in range(m//2)]
            prop_trajectories = [self.rollout(prop_dist, d) for _ in range(m//2)]
            trajectories = nom_trajectories + prop_trajectories
            
            # Exit early if no trajectories
            if not trajectories:
                return 0.0, 0.0
                
            # compute log weights directly
            log_nom = np.array([self.logpdf(pnom, traj) for traj in trajectories])
            log_prop = np.array([self.logpdf(prop_dist, traj) for traj in trajectories])
            log_weights = log_nom - log_prop
            
            # apply more stable transformation to log weights
            # log_weights = self.stabilize_log_weights(log_weights)
            
            # proceed with stabilization and normalization
            max_log_weight = np.max(log_weights)
            stabilized_weights = np.exp(log_weights - max_log_weight)
            
            # ensure weights are properly normalized
            if np.sum(stabilized_weights) > 1e-10:
                normalized_weights = stabilized_weights / np.sum(stabilized_weights)
            else:
                # fall back to uniform weights if numerical issues occur
                normalized_weights = np.ones_like(stabilized_weights) / len(stabilized_weights)
            
            # compute weighted average of samples from the proposal distribution
            failure_indicators = np.array([self.is_failure(trajectory) for trajectory in trajectories])
            
            # additional sanity check before final calculation
            if np.sum(failure_indicators) > 0:
                # the weighted estimate is simply the sum of normalized weights for failure cases
                failure_probability = np.sum(normalized_weights[failure_indicators])
            else:
                failure_probability = 0.0
            
            # calcualte variance and standard error
            if failure_probability > 0:
                variance = np.sum(normalized_weights**2 * (failure_indicators - failure_probability)**2)
                std_error = np.sqrt(variance)
            else:
                # if prob is 0, estimate standard error based on sample size
                std_error = np.sqrt((0 * (1-0)) / len(trajectories))
            
            failure_probs.append(failure_probability)
            std_errors.append(std_error)

        # final failure probability and standard error
        final_failure_prob = np.mean(failure_probs)
        final_std_error = np.sqrt(np.mean(np.array(std_errors)**2))

        return final_failure_prob, final_std_error