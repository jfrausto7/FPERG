import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math

from policies.DQN.DQN import DQN
from policies.DQN.ReplayMemory import ReplayMemory, Transition

class DQNGraspingPolicy:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        print(f"Using device: {self.device}")
        
        # Initialize neural networks
        self.policy_net = DQN().to(device)  # Network for selecting actions
        self.target_net = DQN().to(device)  # Stable network for computing target values
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in evaluation mode
        
        # Exploration parameters
        self.eps_start = 1.0        # Start with 100% exploration
        self.eps_end = 0.3          # Maintain 20% exploration at minimum
        self.eps_decay = 3000      # Slow decay for thorough exploration
        
        # Learning parameters
        self.batch_size = 64       # Larger batches for stable learning
        self.gamma = 0.95            # Discount factor for future rewards
        self.learning_rate = 5e-4   # Conservative learning rate
        self.target_update = 10     # Update target network every 10 episodes
        
        # Memory management
        self.memory = ReplayMemory(50000)  # Store more experiences for better sampling
        
        # Action parameters
        self.action_step = 0.03     # Small steps for precise control
        self.angle_step = 0.15      # Small angle changes
        
        # Initialize optimizer with stability improvements
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5  # L2 regularization to prevent overfitting
        )
        
        # Training progress tracking
        self.steps_done = 0
        self.episode_rewards = []
        self.successful_grasps = 0

    def reset(self):
        """Reset episode-specific variables"""
        pass

    def preprocess_observation(self, observation):
        """Convert numpy observation to PyTorch tensor"""
        return torch.FloatTensor(observation).unsqueeze(0).to(self.device)

    def select_action(self, observation, training=True):
        """
        Select action using epsilon-greedy policy with decaying exploration
        Returns continuous action vector [dx, dy, dz, da]
        """
        state = self.preprocess_observation(observation)
        
        if training:
            # Calculate exploration rate
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
            self.steps_done += 1
            
            # Epsilon-greedy action selection
            if random.random() > eps_threshold:
                with torch.no_grad():  # Don't track gradients for action selection
                    action_idx = self.policy_net(state).max(1)[1].view(1, 1)
            else:
                action_idx = torch.tensor([[random.randrange(7)]], 
                                        device=self.device, dtype=torch.long)
        else:
            # During evaluation, always choose best action
            with torch.no_grad():
                action_idx = self.policy_net(state).max(1)[1].view(1, 1)
        
        # Convert discrete action index to continuous action vector
        dx = [0, -0.05, 0.05, 0, 0, 0, 0][action_idx.item()]
        dy = [0, 0, 0, -0.05, 0.05, 0, 0][action_idx.item()]
        dz = -0.04  # Slightly slower descent
        da = [0, 0, 0, 0, 0, -0.2, 0.2][action_idx.item()]
        
        return np.array([dx, dy, dz, da])

    def update(self, state, action, reward, next_state, done):
        """
        Store transition in memory and perform optimization step
        Implements experience replay and weighted learning
        """
        # Convert continuous action back to discrete index
        dx, dy, dz, da = action
        
        # Action conversion lookup
        if dx == -self.action_step:
            discrete_action = 1
        elif dx == self.action_step:
            discrete_action = 2
        elif dy == -self.action_step:
            discrete_action = 3
        elif dy == self.action_step:
            discrete_action = 4
        elif da == -self.angle_step:
            discrete_action = 5
        elif da == self.angle_step:
            discrete_action = 6
        else:
            discrete_action = 0
            
        # Store transition in replay memory
        state_tensor = self.preprocess_observation(state)
        next_state_tensor = self.preprocess_observation(next_state) if not done else None
        
        self.memory.push(
            state_tensor,
            torch.tensor([[discrete_action]], device=self.device),
            next_state_tensor,
            torch.tensor([reward], device=self.device)
        )
        
        # Perform optimization if we have enough samples
        self.optimize_model()

    def optimize_model(self):
        """
        Perform one step of optimization on the DQN
        Uses weighted learning and gradient clipping for stability
        """
        if len(self.memory) < self.batch_size:
            return
            
        # Sample transitions from memory
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Create mask for non-terminal states
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        
        # Concatenate batch elements
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute current Q-values
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        
        # Compute expected Q-values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute weighted Huber loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1),
            reduction='none'
        )
        
        # Weight losses based on rewards to emphasize successful experiences
        weights = torch.exp(reward_batch / 100.0)
        weighted_loss = (loss * weights.unsqueeze(1)).mean()

        # Optimize the model with gradient clipping
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def save_policy(self, filename):
        """Save policy network and optimizer state"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'successful_grasps': self.successful_grasps
        }, filename)

    def load_policy(self, filename):
        """Load policy network and optimizer state"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint.get('steps_done', 0)
        self.successful_grasps = checkpoint.get('successful_grasps', 0)