from environment.GraspEnv import GraspEnv
from policies.DQN.DQNGraspingPolicy import DQNGraspingPolicy
from policies.GraspingPolicy import GraspingPolicy
from policies.HillClimbingGraspingPolicy import HillClimbingGraspingPolicy 
import numpy as np
import argparse
from collections import deque
import time
import os

def train_basic_policy(n_episodes=100, gui=False):
    """Train using basic grasping policy - mainly for baseline evaluation
    since this policy doesn't actually learn/improve"""
    env = GraspEnv(gui=gui)
    policy = GraspingPolicy()
    
    # Training metrics
    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    
    print("Evaluating basic policy performance...")
    for episode in range(n_episodes):
        obs = env.reset()
        policy.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = policy.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        # Update metrics
        success_history.append(float(info['grasp_success']))
        reward_history.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode % 100 == 0 and episode != 0:
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate (last 100): {np.mean(success_history):.2%}")
            print(f"Average Reward (last 100): {np.mean(reward_history):.1f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
    
    env.cleanup()
    return np.mean(success_history)

def train_hill_climbing(n_episodes=2000, gui=False, load_existing=False):
    """Train using hill climbing policy"""
    env = GraspEnv(gui=gui)
    policy = HillClimbingGraspingPolicy()
    
    if load_existing:
        try:
            policy.load_policy('models/best_hill_climbing_policy.pkl')
            print("Loaded existing hill climbing policy...")
        except:
            print("No existing hill climbing policy found, starting fresh...")

    # Training metrics
    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_success_rate = 0.0
    
    print("Training hill climbing policy...")
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
        
        # Update metrics
        success_history.append(float(info['grasp_success']))
        reward_history.append(episode_reward)
        episode_lengths.append(steps)
        
        # Save best policy
        current_success_rate = np.mean(success_history)
        if current_success_rate > best_success_rate and episode >= 100:
            best_success_rate = current_success_rate
            policy.save_policy('models/best_hill_climbing_policy.pkl')
            print(f"\nNew best policy saved with success rate: {best_success_rate:.2%}")
        
        if episode % 100 == 0 and episode != 0:
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate (last 100): {current_success_rate:.2%}")
            print(f"Average Reward (last 100): {np.mean(reward_history):.1f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
            print(f"Best Success Rate So Far: {best_success_rate:.2%}")
    
    env.cleanup()
    return best_success_rate

def train_dqn(n_episodes=2000, gui=False, load_existing=False):
    """Train using DQN policy"""
    env = GraspEnv(gui=gui)
    policy = DQNGraspingPolicy()
    
    if load_existing:
        try:
            policy.load_policy('models/best_dqn_policy.pt')
            print("Loaded existing DQN policy...")
        except:
            print("No existing DQN policy found, starting fresh...")

    # Training metrics
    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_success_rate = 0.0
    
    print("Training DQN policy...")
    for episode in range(n_episodes):
        obs = env.reset()
        policy.reset()  # Clear frame history
        episode_reward = 0
        steps = 0
        
        while True:
            action = policy.select_action(obs, training=True)
            next_obs, reward, done, info = env.step(action)
            
            # Store transition and optimize
            policy.update(obs, action, reward, next_obs, done)

            if steps % 100 == 0:
                policy.target_net.load_state_dict(policy.policy_net.state_dict())
            
            episode_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        # Update metrics
        success_history.append(float(info['grasp_success']))
        reward_history.append(episode_reward)
        episode_lengths.append(steps)
        
        # Save best policy
        current_success_rate = np.mean(success_history)
        if current_success_rate > best_success_rate and episode >= 100:
            best_success_rate = current_success_rate
            policy.save_policy('models/best_dqn_policy.pt')
            print(f"\nNew best policy saved with success rate: {best_success_rate:.2%}")
        
        if episode % 100 == 0 and episode != 0:
            print(f"\nEpisode {episode}/{n_episodes}")
            print(f"Success Rate (last 100): {current_success_rate:.2%}")
            print(f"Average Reward (last 100): {np.mean(reward_history):.1f}")
            print(f"Average Episode Length: {np.mean(episode_lengths):.1f}")
            print(f"Best Success Rate So Far: {best_success_rate:.2%}")
            
            # Update target network periodically
            policy.target_net.load_state_dict(policy.policy_net.state_dict())
    
    env.cleanup()
    return best_success_rate

def evaluate_policy(policy_type, n_episodes=100, gui=False):
    """Evaluate a trained policy"""
    env = GraspEnv(gui=gui)
    
    # Load appropriate policy
    if policy_type == 'basic':
        policy = GraspingPolicy()
    elif policy_type == 'hill':
        policy = HillClimbingGraspingPolicy()
        try:
            policy.load_policy('models/best_hill_climbing_policy.pkl')
        except:
            print("No hill climbing policy file found! Please train first.")
            return None
    elif policy_type == 'dqn':
        policy = DQNGraspingPolicy()
        try:
            policy.load_policy('models/best_dqn_policy.pt')
        except:
            print("No DQN policy file found! Please train first.")
            return None
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    # Evaluation metrics
    successes = 0
    rewards = []
    episode_lengths = []
    
    print(f"\nEvaluating {policy_type} policy...")
    for episode in range(n_episodes):
        obs = env.reset()
        policy.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Use appropriate action selection method
            if policy_type == 'dqn':
                action = policy.select_action(obs, training=False)
            else:
                action = policy.get_action(obs)
                
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if done:
                break
        
        successes += info['grasp_success']
        rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        if episode % 100 == 0:
            print(f"Completed {episode}/{n_episodes} evaluation episodes...")
            print(f"Current success rate: {successes/(episode+1):.2%}")
    
    # Final results
    success_rate = successes/n_episodes
    print(f"\nEvaluation Results for {policy_type} policy:")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Average reward: {np.mean(rewards):.1f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    
    env.cleanup()
    return success_rate

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate grasping policies')
    parser.add_argument('--policy', type=str, default='hill',
                      choices=['basic', 'hill', 'dqn'],
                      help='Policy type to use (default: hill)')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'eval'],
                      help='Whether to train or evaluate (default: train)')
    parser.add_argument('--gui', action='store_true', default=False,
                      help='Use GUI mode instead of DIRECT mode')
    parser.add_argument('--episodes', type=int, default=None,
                      help='Number of episodes (default: policy-dependent)')
    parser.add_argument('--load', action='store_true', default=False,
                      help='Load existing policy')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Set default episodes based on policy type
    if args.episodes is None:
        if args.policy == 'basic':
            args.episodes = 100
        elif args.policy == 'hill':
            args.episodes = 2000
        else:  # dqn
            args.episodes = 2000
    
    # Record start time
    start_time = time.time()
    
    # Run appropriate training or evaluation
    if args.mode == 'train':
        print(f"\nTraining {args.policy} policy for {args.episodes} episodes...")
        if args.policy == 'basic':
            success_rate = train_basic_policy(args.episodes, args.gui)
        elif args.policy == 'hill':
            success_rate = train_hill_climbing(args.episodes, args.gui, args.load)
        else:  # dqn
            success_rate = train_dqn(args.episodes, args.gui, args.load)
    else:  # eval
        print(f"\nEvaluating {args.policy} policy...")
        success_rate = evaluate_policy(args.policy, args.episodes, args.gui)
    
    # Print timing information
    elapsed_time = time.time() - start_time
    print(f"\nTotal time: {elapsed_time:.1f} seconds")
    print(f"Average time per episode: {elapsed_time/args.episodes:.1f} seconds")

if __name__ == "__main__":
    main()