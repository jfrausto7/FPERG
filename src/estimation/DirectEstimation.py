import numpy as np
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
import pandas as pd
import pybullet as p
from typing import Dict, List, Tuple
import time

Z_SCORE_95_CI = 1.96

class DirectEstimation:
    def __init__(self, n_trials: int = 1000, gui: bool = False, use_hill_climbing: bool = False, policy_file: str = None):
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
        
    def run_trial(self) -> Dict:
        """Run a single trial and return the results."""
        obs = self.env.reset()
        self.policy.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # record initial state
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.env.object_id)
        initial_state = {
            'obj_pos_x': obj_pos[0],
            'obj_pos_y': obj_pos[1],
            'obj_pos_z': obj_pos[2],
            'obj_orn_x': obj_orn[0],
            'obj_orn_y': obj_orn[1],
            'obj_orn_z': obj_orn[2],
            'obj_orn_w': obj_orn[3]
        }
        
        while not done:
            action = self.policy.get_action(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            if self.gui:
                time.sleep(1./120.)
        
        # combine results
        result = {
            'success': info['grasp_success'],
            'total_reward': total_reward,
            'steps': steps,
            'lift_duration': info['lift_duration'],
            **initial_state  # unpack initial state
        }
        
        return result
    
    def estimate_failure_probability(self) -> Tuple[float, float, pd.DataFrame]:
        """
        Run trials and estimate failure probability.
        
        Returns:
            a tuple containing:
            - estimated failure probability
            - standard error of the estimate
            - a DataFrame with detailed results
        """
        print(f"Starting direct estimation with {self.n_trials} trials...")
        
        for i in range(self.n_trials):
            if i % 50 == 0:
                print(f"Completed {i}/{self.n_trials} trials...")
            
            result = self.run_trial()
            self.results.append(result)
        
        
        # calculate failure probability and standard error
        df = pd.DataFrame(self.results)
        successes = df['success'].sum()
        failures = self.n_trials - successes
        p_failure = failures / self.n_trials
        
        # standard error for a binomial proportion
        std_error = np.sqrt((p_failure * (1 - p_failure)) / self.n_trials)
        
        print(f"\nResults:")
        print(f"Failure Probability: {p_failure:.4f} ± {std_error:.4f}")
        print(f"Based on {failures} failures in {self.n_trials} trials")
        
        return p_failure, std_error, df
    
    def analyze_failure_modes(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns in failure cases."""
        failure_cases = df[~df['success']]
        success_cases = df[df['success']]
        
        analysis = {
            'total_trials': len(df),
            'failure_count': len(failure_cases),
            'failure_rate': len(failure_cases) / len(df),
            'avg_steps_failure': failure_cases['steps'].mean(),
            'avg_steps_success': success_cases['steps'].mean(),
            'avg_reward_failure': failure_cases['total_reward'].mean(),
            'avg_reward_success': success_cases['total_reward'].mean(),
            # position analysis
            'failed_x_mean': failure_cases['obj_pos_x'].mean(),
            'failed_y_mean': failure_cases['obj_pos_y'].mean(),
            'failed_x_std': failure_cases['obj_pos_x'].std(),
            'failed_y_std': failure_cases['obj_pos_y'].std(),
        }
        
        return analysis
    
    def save_results(self, df: pd.DataFrame, filename: str = 'direct_estimation_results.csv'):
        """
        Save results to CSV file with metadata header.
        
        Args:
            df: DataFrame containing trial results
            filename: Path to save results
        """
        # calculate aggregate statistics
        successes = df['success'].sum()
        failures = len(df) - successes
        p_failure = failures / len(df)
        std_error = np.sqrt((p_failure * (1 - p_failure)) / len(df))
        
        # create metadata
        metadata = pd.DataFrame([
            ['n_trials', len(df)],
            ['failures', failures],
            ['successes', successes],
            ['failure_probability', p_failure],
            ['standard_error', std_error],
            ['confidence_interval_low', p_failure - Z_SCORE_95_CI * std_error],
            ['confidence_interval_high', p_failure + Z_SCORE_95_CI * std_error],
            ['timestamp', time.strftime("%Y-%m-%d %H:%M:%S")]
        ], columns=['metric', 'value'])
        
        # save both metadata & results
        with open(filename, 'w') as f:
            f.write("# Experiment Metadata\n")
            metadata.to_csv(f, index=False)
            f.write("\n# Trial Results\n")
            df.to_csv(f, index=False)
            
        print(f"Results saved to {filename}")
    
    def cleanup(self):
        """Cleanup environment."""
        self.env.cleanup()

def main():
    # run direct estimation
    estimator = DirectEstimation(n_trials=1000, gui=False)
    p_failure, std_error, results_df = estimator.estimate_failure_probability()
    
    # analyze failure modes
    analysis = estimator.analyze_failure_modes(results_df)
    
    # print detailed analysis
    print("\nFailure Mode Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value:.4f}")
    
    # save results
    estimator.save_results(results_df)
    estimator.cleanup()

if __name__ == "__main__":
    main()