from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
import argparse
import time
import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path

SAVE_PATH = "src/data/"

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
    actions_df = DataFrame([], columns=["attempt", "step", "observation", "action", "reward", "done", "info" ])
    actions_df_index = 0
    attempts_df = DataFrame([], columns=["attempt", "episode reward", "success"])
    attempts_df_index = 0

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
        
        step = 0
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            actions_df.loc[actions_df_index] = [i, step, obs, action, reward, done, info]
            actions_df_index += 1
            step += 1
            if gui_mode:
                time.sleep(1./120.)
            
        attempts_df.loc[attempts_df_index] = [i, episode_reward, info['grasp_success']]
        attempts_df_index += 1
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
    curr_time = str(time.time_ns())
    actions_filename = SAVE_PATH + "actions_from_run_at_" + curr_time + ".csv"
    attempts_filename = SAVE_PATH + "attempts_from_run_at_" + curr_time + ".csv"
    actions_df.to_csv(actions_filename)
    attempts_df.to_csv(attempts_filename)
    
    env.cleanup()
    return final_success_rate

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
    
    args = parser.parse_args()
    
    # set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # print selected config
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
