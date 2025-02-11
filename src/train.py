from environment.GraspEnv import GraspEnv
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy
import numpy as np
import argparse
from collections import deque

def train_hill_climbing(n_episodes=2000, gui=False, load_existing=False):
    """Train using hill climbing policy"""
    env = GraspEnv(gui=gui)
    policy = HillClimbingGraspingPolicy()
    
    if load_existing:
        try:
            policy.load_policy('src/best_hill_climbing_policy.pkl')
            print("Loaded existing policy...")
        except:
            print("No existing policy found, starting fresh...")
    
    # training metrics
    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    for episode in range(n_episodes):
        obs = env.reset()
        policy.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            policy.update(obs, action, reward, next_obs, done, info)
            
            episode_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        # update metrics
        success_history.append(float(info['grasp_success']))
        reward_history.append(episode_reward)
        episode_lengths.append(steps)
        
        # calculate current performance
        current_success_rate = np.mean(success_history)
        current_reward = np.mean(reward_history)
        
        if episode % 100 == 0 and episode != 0:
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate (last 100): {current_success_rate:.2%}")
            print(f"Average Reward (last 100): {current_reward:.1f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
            print(f"Best Success Rate: {policy.best_success_rate:.2%}")
    
    # save best policy
    policy.save_policy('src/best_hill_climbing_policy.pkl')
    env.cleanup()
    return policy

def evaluate_policy(policy, n_episodes=100, gui=False):
    """Evaluate a trained policy"""
    env = GraspEnv(gui=gui)
    successes = 0
    rewards = []
    
    # use best parameters for evaluation
    policy.current_params = policy.best_params
    
    for episode in range(n_episodes):
        obs = env.reset()
        policy.reset()
        episode_reward = 0
        
        while True:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        successes += info['grasp_success']
        rewards.append(episode_reward)
        
        if episode % 10 == 0:
            print(f"Completed {episode}/{n_episodes} evaluation episodes...")
            print(f"Current success rate: {successes/(episode+1):.2%}")
    
    print("\nEvaluation Results:")
    print(f"Success rate: {successes/n_episodes:.2%}")
    print(f"Average reward: {np.mean(rewards):.1f}")
    print(f"Std reward: {np.std(rewards):.1f}")
    
    env.cleanup()
    return successes/n_episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train hill climbing grasping policy')
    parser.add_argument('--gui', action='store_true', default=False,
                      help='Use GUI mode instead of DIRECT mode')
    parser.add_argument('--episodes', type=int, default=2000,
                      help='Number of training episodes')
    parser.add_argument('--load', action='store_true', default=False,
                      help='Load existing policy')
    parser.add_argument('--eval', action='store_true', default=False,
                      help='Evaluate only (no training)')
    
    args = parser.parse_args()
    
    if args.eval:
        policy = HillClimbingGraspingPolicy()
        try:
            policy.load_policy('src/best_hill_climbing_policy.pkl')
            evaluate_policy(policy, n_episodes=100, gui=args.gui)
        except:
            print("No policy file found! Please train first.")
    else:
        policy = train_hill_climbing(args.episodes, args.gui, args.load)
        evaluate_policy(policy, n_episodes=100, gui=False)