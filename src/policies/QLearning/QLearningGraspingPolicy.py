import numpy as np
from collections import defaultdict
import pickle

class QLearningGraspingPolicy:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()))
        
        # discretization parameters
        self.pos_bins = 10
        self.dist_bins = 10
        self.gripper_states = 2  # ppen/Closed
        
        # action space definition
        self.dx_options = [-0.05, 0, 0.05]  # small discrete movements
        self.dy_options = [-0.05, 0, 0.05]
        self.dz_options = [-0.05, 0, 0.05]
        self.finger_options = [-1, 1]
        
    def get_action_space_size(self):
        """Calculate total number of discrete actions"""
        return len(self.dx_options) * len(self.dy_options) * \
               len(self.dz_options) * len(self.finger_options)
    
    def discretize_state(self, obs):
        """convert continuous state to discrete state for Q-table"""
        gripper_pos = obs[0:3]
        object_pos = obs[3:6]
        
        # calculate relevant features
        xy_dist = np.sqrt((object_pos[0] - gripper_pos[0])**2 + 
                         (object_pos[1] - gripper_pos[1])**2)
        z_dist = gripper_pos[2] - object_pos[2]
        
        # discretize features
        xy_dist_disc = np.digitize([xy_dist], np.linspace(0, 0.5, self.dist_bins))[0]
        z_dist_disc = np.digitize([z_dist], np.linspace(-0.2, 0.4, self.dist_bins))[0]
        x_pos_disc = np.digitize([gripper_pos[0]], np.linspace(0.3, 0.7, self.pos_bins))[0]
        y_pos_disc = np.digitize([gripper_pos[1]], np.linspace(-0.3, 0.3, self.pos_bins))[0]
        z_pos_disc = np.digitize([gripper_pos[2]], np.linspace(-0.2, 0.5, self.pos_bins))[0]
        
        # combine into discrete state
        return (xy_dist_disc, z_dist_disc, x_pos_disc, y_pos_disc, z_pos_disc)
    
    def action_to_continuous(self, action_idx):
        """Convert discrete action index to continuous action vector"""
        n_dx = len(self.dx_options)
        n_dy = len(self.dy_options)
        n_dz = len(self.dz_options)
        
        # decode action index into individual components
        finger_idx = action_idx % len(self.finger_options)
        temp = action_idx // len(self.finger_options)
        dz_idx = temp % len(self.dz_options)
        temp = temp // len(self.dz_options)
        dy_idx = temp % len(self.dy_options)
        dx_idx = temp // len(self.dy_options)
        
        # convert to continuous actions
        return np.array([
            self.dx_options[dx_idx],
            self.dy_options[dy_idx],
            self.dz_options[dz_idx],
            self.finger_options[finger_idx]
        ])
    
    def get_action(self, obs):
        """Select action using epsilon-greedy policy"""
        state = self.discretize_state(obs)
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(self.get_action_space_size())
        else:
            action_idx = np.argmax(self.q_table[state])
        
        return self.action_to_continuous(action_idx)
    
    def update(self, obs, action, reward, next_obs):
        """Update Q-value using Q-learning update rule"""
        state = self.discretize_state(obs)
        next_state = self.discretize_state(next_obs)
        
        # convert continuous action back to discrete index (this is approximate - we find the closest discrete action)
        action_diffs = np.array([
            np.sum(np.abs(self.action_to_continuous(i) - action))
            for i in range(self.get_action_space_size())
        ])
        action_idx = np.argmin(action_diffs)
        
        # Q-learning update
        best_next_value = np.max(self.q_table[next_state])
        current_value = self.q_table[state][action_idx]
        self.q_table[state][action_idx] = current_value + self.lr * (
            reward + self.gamma * best_next_value - current_value
        )
    
    def save_policy(self, filename='q_policy.pkl'):
        """Save the Q-table to a file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_policy(self, filename='q_policy.pkl'):
        """Load the Q-table from a file"""
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(self.get_action_space_size()),
                                     pickle.load(f))
            
    def decay_epsilon(self, decay_rate=0.995):
        """Decay exploration rate"""
        self.epsilon *= decay_rate
        self.epsilon = max(0.01, self.epsilon)  # Don't let it get too low lol