from environment.GraspEnv import GraspEnv
from policies.GraspingPolicy import GraspingPolicy
import argparse
import time
import numpy as np

def run_single_grasp(gui_mode=False):
    """Run a single grasp attempt"""
    env = GraspEnv(gui=gui_mode)
    obs = env.reset()
    done = False
    total_reward = 0
    policy = GraspingPolicy()
    
    while not done:
        action = policy.get_action(obs)
        # print(action)
        obs, reward, done, info = env.step(action)
        # print(obs, reward, done, info)
        total_reward += reward
        
        if gui_mode:
            time.sleep(1./240.)  # Slow down for visualization
    
    success = info['grasp_success']
    print(f"Grasp {'succeeded' if success else 'failed'} with reward {total_reward:.2f}")
    env.cleanup()
    return success

def run_multiple_grasps(n_attempts=100, gui_mode=False):
    """Run multiple grasp attempts and report success rate"""
    env = GraspEnv(gui=gui_mode)
    successes = 0
    total_rewards = []
    policy = GraspingPolicy()
    
    for i in range(n_attempts):
        obs = env.reset()
        done = False
        episode_reward = 0
        policy.reset()  # Reset policy state for new attempt
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if gui_mode:
                time.sleep(1./240.)
        
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

def main():
    parser = argparse.ArgumentParser(description='Run robotic grasping experiments')
    parser.add_argument('--gui', action='store_true', default=False,
                      help='Use GUI mode instead of DIRECT mode (default: DIRECT)')
    parser.add_argument('--multiple', type=int, metavar='N', default=None,
                      help='Run N grasp attempts (default: None, runs single grasp)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed (default: None)')
    
    args = parser.parse_args()
    
    # set random seed if provided (for experiments and data collection)
    if args.seed is not None:
        np.random.seed(args.seed)
        
    if args.multiple is not None:
        print(f"Running {args.multiple} grasp attempts in {'GUI' if args.gui else 'DIRECT'} mode...")
        run_multiple_grasps(args.multiple, args.gui)
    else:
        print(f"Running single grasp in {'GUI' if args.gui else 'DIRECT'} mode...")
        run_single_grasp(args.gui)

if __name__ == "__main__":
    main()