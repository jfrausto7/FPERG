import os
from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
from estimation.DirectEstimation import DirectEstimation
import argparse
import time
import numpy as np

def run_single_grasp(gui_mode=False, use_qlearning=False, policy_file=None):
    """Run a single grasp attempt"""
    env = GraspEnv(gui=gui_mode)
    obs = env.reset()
    done = False
    total_reward = 0
    
    # initialize appropriate policy
    if use_qlearning:
        policy = HillClimbingGraspingPolicy()
        if policy_file:
            try:
                policy.load_policy(policy_file)
                print(f"Loaded Q-learning policy from {policy_file}")
            except:
                print(f"Warning: Could not load policy from {policy_file}")
        policy.epsilon = 0  # disable exploration for evaluation
    else:
        policy = GraspingPolicy()
    
    while not done:
        action = policy.get_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if gui_mode:
            time.sleep(1./120.)
    
    success = info['grasp_success']
    print(f"Grasp {'succeeded' if success else 'failed'} with reward {total_reward:.2f}")
    env.cleanup()
    return success

def run_multiple_grasps(n_attempts=100, gui_mode=False, use_qlearning=False, policy_file=None):
    """Run multiple grasp attempts and report success rate"""
    env = GraspEnv(gui=gui_mode)
    successes = 0
    total_rewards = []
    
    # Initialize appropriate policy
    if use_qlearning:
        policy = HillClimbingGraspingPolicy()
        if policy_file:
            try:
                policy.load_policy(policy_file)
                print(f"Loaded Q-learning policy from {policy_file}")
            except:
                print(f"Warning: Could not load policy from {policy_file}")
        policy.epsilon = 0
    else:
        policy = GraspingPolicy()
    
    for i in range(n_attempts):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        if hasattr(policy, 'reset'):  # reset policy state if method exists
            policy.reset()
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if gui_mode:
                time.sleep(1./120.)
        
        successes += info['grasp_success']
        total_rewards.append(episode_reward)
        
        if i % 10 == 0:
            print(f"Completed {i}/{n_attempts} attempts...")
            print(f"Current success rate: {successes/(i+1):.2%}")
            print(f"Average reward: {np.mean(total_rewards):.2f}")
    
    final_success_rate = successes/n_attempts
    print(f"\nFinal Results:")
    print(f"Success rate: {final_success_rate:.2%}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Std reward: {np.std(total_rewards):.2f}")
    
    env.cleanup()
    return final_success_rate

def run_direct_estimation(n_trials=1000, gui_mode=False, use_hill_climbing=False, policy_file=None):
    """Run direct estimation of failure probability"""
    print(f"\nRunning direct estimation with {n_trials} trials...")
    print(f"Using {'hill climbing' if use_hill_climbing else 'default'} policy")

    # create results directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    estimator = DirectEstimation(
        n_trials=n_trials, 
        gui=gui_mode,
        use_hill_climbing=use_hill_climbing,
        policy_file=policy_file
    )
    
    try:
        # Run estimation
        p_failure, std_error, results_df = estimator.estimate_failure_probability()
        
        # Analyze results
        analysis = estimator.analyze_failure_modes(results_df)
        
        # Save results with policy type in filename
        policy_type = 'hill_climbing' if use_hill_climbing else 'default'
        filename = f'results/direct_estimation_results_{policy_type}.csv'
        estimator.save_results(results_df, filename)
        
    finally:
        estimator.cleanup()
    
    return p_failure, std_error

def main():
    parser = argparse.ArgumentParser(description='Run robotic grasping experiments')
    parser.add_argument('--gui', action='store_true', default=False,
                      help='Use GUI mode instead of DIRECT mode (default: DIRECT)')
    parser.add_argument('--multiple', type=int, metavar='N', default=None,
                      help='Run N grasp attempts (default: None, runs single grasp)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: None)')
    parser.add_argument('--hill', action='store_true', default=False,
                      help='Use Hill Climbing policy instead of default policy')
    parser.add_argument('--policy-file', type=str, default='src/best_hill_climbing_policy.pkl',
                      help='Path to Q-learning policy file (default: best_hill_climbing_policy.pkl)')
    parser.add_argument('--estimate', action='store_true', default=False,
                      help='Run direct estimation of failure probability')
    parser.add_argument('--trials', type=int, default=1000,
                      help='Number of trials for direct estimation (default: 1000)')
    
    args = parser.parse_args()
    
    # set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    if args.estimate:
        print("Running direct estimation of failure probability...")
        p_failure, std_error = run_direct_estimation(
            args.trials, 
            args.gui,
            args.hill,  # Pass hill climbing flag
            args.policy_file if args.hill else None  # Pass policy file only if using hill climbing
        )
        print(f"\nFinal Results:")
        print(f"Failure Probability: {p_failure:.4f} ± {std_error:.4f}")
        print(f"95% Confidence Interval: [{p_failure - 1.96*std_error:.4f}, {p_failure + 1.96*std_error:.4f}]")
        return
    
    # print selected config for regular runs
    mode = 'GUI' if args.gui else 'DIRECT'
    policy_type = 'Hill Climbing' if args.hill else 'Default'
    print(f"Running with {policy_type} policy in {mode} mode...")
    if args.hill:
        print(f"Using policy file: {args.policy_file}")
        
    if args.multiple is not None:
        print(f"Running {args.multiple} grasp attempts...")
        run_multiple_grasps(args.multiple, args.gui, args.hill, args.policy_file)
    else:
        print("Running single grasp...")
        run_single_grasp(args.gui, args.hill, args.policy_file)

if __name__ == "__main__":
    main()