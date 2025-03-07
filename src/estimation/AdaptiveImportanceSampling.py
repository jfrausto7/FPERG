import numpy as np
import math
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
from distributions.nominal_trajectory_dist import NominalTrajectoryDistribution
from distributions.proposal_trajectory_dist import ProposalTrajectoryDistribution
from typing import Dict, List
import matplotlib.pyplot as plt
import os

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
            new_q = ProposalTrajectoryDistribution(float(weighted_mean[0]), float(avg_std), q.depth)
            return new_q
        
        # slightly widen the current distribution when there are no valid disturbances
        return ProposalTrajectoryDistribution(q.mean, q.std * 1.1, q.depth)

    """
    Iterates to find a proposal distribution for importance sampling.
    Returns the final proposal distribution and history of distributions.
    """
    def find_proposal_dist(self, nominal_dist, proposal_dist, f, k_max, m, m_elite, d):
        # Track the distribution history
        dist_history = [proposal_dist]
        
        for k in range(k_max):
            print(f"Iteration {k+1}/{k_max}")
            # Perform rollouts with proposal
            trajectories = [self.rollout(proposal_dist, d) for _ in range(m)]
            Y = [f(traj) for traj in trajectories]
            Y.sort()
            cutoff = max(0, Y[m_elite] if m_elite < len(Y) else 0)
            ps = np.array([self.logpdf(nominal_dist, traj) for traj in trajectories])
            qs = np.array([self.logpdf(proposal_dist, traj) for traj in trajectories])
            ws = ps / qs
            ws = [w if y < cutoff else 0 for w,y in zip(ws,Y)]
            proposal_dist = self.fit(proposal_dist, trajectories, ws)
            dist_history.append(proposal_dist)
            
            print(f"  Updated proposal: mean={proposal_dist.mean:.3f}, std={proposal_dist.std:.3f}")
            
        return proposal_dist, dist_history

    def plot_distributions(self, nominal_dist, proposal_dist, save_path=None):
        """
        Plot the nominal and proposal distributions after AIS is complete.
        
        Args:
            nominal_dist: The nominal trajectory distribution
            proposal_dist: The adapted proposal distribution
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Nominal vs Proposal Distributions After Adaptive Importance Sampling', fontsize=16)
        
        # initial state distributions
        x_nominal_dist, y_nominal_dist = nominal_dist.initial_state_distribution()
        x_proposal_dist, y_proposal_dist = proposal_dist.initial_state_distribution()
        
        # create x ranges for plotting uniform distributions
        x_range = np.linspace(0.4, 0.6, 1000)
        y_range = np.linspace(-0.1, 0.05, 1000)
        
        # plot X position distribution
        axes[0, 0].plot(x_range, x_nominal_dist.pdf(x_range), 'b-', label='Nominal')
        axes[0, 0].plot(x_range, x_proposal_dist.pdf(x_range), 'r-', label='Proposal')
        axes[0, 0].set_title('Object X Position Distribution')
        axes[0, 0].set_xlabel('X Position')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # plot Y position distribution
        axes[0, 1].plot(y_range, y_nominal_dist.pdf(y_range), 'b-', label='Nominal')
        axes[0, 1].plot(y_range, y_proposal_dist.pdf(y_range), 'r-', label='Proposal')
        axes[0, 1].set_title('Object Y Position Distribution')
        axes[0, 1].set_xlabel('Y Position')
        axes[0, 1].set_ylabel('Probability Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # plot disturbance distributions
        x_disturbance = np.linspace(-3, 3, 1000)
        nominal_disturb = nominal_dist.disturbance_distribution(0)
        proposal_disturb = proposal_dist.disturbance_distribution(0)
        
        axes[1, 0].plot(x_disturbance, nominal_disturb.pdf(x_disturbance), 'b-', 
                       label=f'Nominal (μ={nominal_dist.mean:.2f}, σ={nominal_dist.std:.2f})')
        axes[1, 0].plot(x_disturbance, proposal_disturb.pdf(x_disturbance), 'r-', 
                       label=f'Proposal (μ={proposal_dist.mean:.2f}, σ={proposal_dist.std:.2f})')
        axes[1, 0].set_title('Disturbance Distribution')
        axes[1, 0].set_xlabel('Disturbance Value')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # plot disturbance cumulative distribution
        axes[1, 1].plot(x_disturbance, nominal_disturb.cdf(x_disturbance), 'b-', label='Nominal')
        axes[1, 1].plot(x_disturbance, proposal_disturb.cdf(x_disturbance), 'r-', label='Proposal')
        axes[1, 1].set_title('Disturbance Cumulative Distribution')
        axes[1, 1].set_xlabel('Disturbance Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution comparison plot saved to {save_path}")
        
        plt.close()

    def plot_distribution_adaptation(self, iteration_dists, save_path=None):
        """
        Plot how the proposal distribution adapts over iterations.
        
        Args:
            iteration_dists: List of proposal distributions at each iteration
            save_path: Optional path to save the figure
        """
        if not iteration_dists:
            print("No iteration distributions provided.")
            return
        
        n_iterations = len(iteration_dists)
        x = np.linspace(-3, 3, 1000)
        
        plt.figure(figsize=(12, 8))
        
        # plot each iteration's disturbance distribution
        for i, dist in enumerate(iteration_dists):
            # color gradient from blue to red
            color = plt.cm.coolwarm(i / (n_iterations-1 if n_iterations > 1 else 1))
            
            disturb_dist = dist.disturbance_distribution(0)
            plt.plot(x, disturb_dist.pdf(x), 
                     color=color, 
                     label=f'Iteration {i+1} (μ={dist.mean:.2f}, σ={dist.std:.2f})')
        
        plt.title('Evolution of Proposal Distribution Over Iterations', fontsize=16)
        plt.xlabel('Disturbance Value', fontsize=12)
        plt.ylabel('Probability Density', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution adaptation plot saved to {save_path}")
        
        plt.close()  # Close to avoid displaying in GUI mode

    """
    Implements adaptive importance sampling to return a failure probability estimation.
    Now includes automatic distribution visualization.
    """
    def adaptiveImportanceSampling(self, d):
        # create results directory if it doesn't exist
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        
        # calculate number of samples
        m = self.n_trials // d
        m_elite = math.ceil(m / 10) 
        # use k_max = 10 for reasonable number of iterations
        k_max = min(10, self.n_trials)  
        f = self.simple_failure_function

        print(f"Running AIS with {self.n_trials} trials, {m} samples per iteration, {k_max} iterations")
        
        pnom = NominalTrajectoryDistribution(d)
        init_prop_dist = ProposalTrajectoryDistribution(0, 0.75, d)
        
        # find optimal proposal distribution and track history
        prop_dist, dist_history = self.find_proposal_dist(pnom, init_prop_dist, f, k_max, m, m_elite, d)

        # perform rollouts with final proposal
        print("Running final estimation with adapted proposal distribution...")
        trajectories = [self.rollout(prop_dist, d) for _ in range(m)]

        # compute log weights directly
        log_nom = np.array([self.logpdf(pnom, traj) for traj in trajectories])
        log_prop = np.array([self.logpdf(prop_dist, traj) for traj in trajectories])
        log_weights = log_nom - log_prop
        stabilized_weights = np.exp(log_weights - np.max(log_weights))  # Stabilize numerically
        normalized_weights = stabilized_weights / np.sum(stabilized_weights)

        # compute weighted average of samples from the proposal distribution
        weighted_samples = [w * self.is_failure(trajectory) for w, trajectory in zip(normalized_weights, trajectories)]
        failure_probability = np.mean(weighted_samples)
        n = len(weighted_samples)
        variance = np.sum((weighted_samples - failure_probability)**2) / (n - 1)
        std_error = np.sqrt(variance / n)   # this should go down over time...

        print("Generating distribution plots...")
        self.plot_distributions(
            pnom, 
            prop_dist, 
            save_path=f"{results_dir}/distribution_comparison.png"
        )
        
        self.plot_distribution_adaptation(
            dist_history, 
            save_path=f"{results_dir}/distribution_adaptation.png"
        )
        
        print(f"Final adapted proposal distribution: mean={prop_dist.mean:.3f}, std={prop_dist.std:.3f}")
        print(f"Estimated failure probability: {failure_probability:.6f}")
        print(f"Standard error: {std_error:.6f}")
        print(f"Distribution plots saved to {results_dir} directory")

        return failure_probability, std_error