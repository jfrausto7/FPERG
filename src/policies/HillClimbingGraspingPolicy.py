import numpy as np
import pickle

class HillClimbingGraspingPolicy:
    def __init__(self):
        # initialize with known good default values
        self.current_params = {
            'approach_height': np.float32(0.22),
            'grasp_correction': np.float32(0.3),
            'lift_correction': np.float32(0.2)
        }
        
        # track best parameters
        self.best_params = self.current_params.copy()
        self.best_success_rate = 0.0
        
        # Hill climbing parameters
        self.current_successes = 0
        self.current_attempts = 0
        self.initial_step_size = 0.01
        self.min_step_size = 0.001
        self.step_size = self.initial_step_size
        self.evaluation_window = 30
        self.patience = 50  # episodes without improvement before reducing step size
        self.episodes_without_improvement = 0
        
        # parameter-specific step sizes and ranges
        self.param_config = {
            'approach_height': {
                'step_size': 0.01,
                'range': (0.20, 0.24),
                'importance': 1.0  # weight for parameter selection (more likely to be selected)
            },
            'grasp_correction': {
                'step_size': 0.02,
                'range': (0.25, 0.35),
                'importance': 1.5
            },
            'lift_correction': {
                'step_size': 0.015,
                'range': (0.15, 0.25),
                'importance': 0.8
            }
        }
        
        # track performance history
        self.success_history = []
        self.parameter_history = []
        
        # action scaling
        self.action_scale = np.float32(0.15)
        
        # keep track of consecutive successes/failures for momentum
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        
        # init state
        self.reset()
        
    def reset(self):
        """Reset episode-specific variables"""
        self.stage = 0
        self.stage_timer = 0
        self.has_approached = False
        self.has_grasped = False
        self.last_action = np.zeros(4, dtype=np.float32)
        
    def _try_new_parameters(self):
        """generate new parameters to test with adaptive step sizes"""
        # select parameter to modify based on importance weights
        weights = [config['importance'] for config in self.param_config.values()]
        param = np.random.choice(list(self.param_config.keys()), p=np.array(weights)/sum(weights))
        
        # copy current best parameters
        new_params = self.best_params.copy()
        
        # calculate adaptive step size based on recent performance
        base_step = self.param_config[param]['step_size'] * (self.step_size / self.initial_step_size)
        
        # add momentum based on consecutive successes/failures
        if self.consecutive_successes > 3:
            # If doing well, make smaller adjustments
            base_step *= 0.7
        elif self.consecutive_failures > 3:
            # If doing poorly, make larger adjustments
            base_step *= 1.3
        
        # add or subtract step size with noise
        direction = 1 if np.random.random() < 0.5 else -1
        noise = np.random.normal(0, 0.2 * base_step)  # add some noise to the step
        new_params[param] += direction * base_step + noise
            
        # clip parameters to configured ranges
        param_range = self.param_config[param]['range']
        new_params[param] = np.clip(new_params[param], param_range[0], param_range[1])
            
        return new_params
    
    def _adjust_step_size(self):
        """Adjust step size based on improvement history"""
        if self.episodes_without_improvement > self.patience:
            self.step_size = max(self.step_size * 0.8, self.min_step_size)
            self.episodes_without_improvement = 0
            print(f"Reducing step size to {self.step_size:.6f}")
    
    def update(self, obs, action, reward, next_obs, done, info):
        """Update policy based on episode results with enhanced tracking"""
        if done:
            success = info.get('grasp_success', False)
            self.current_attempts += 1
            
            if success:
                self.current_successes += 1
                self.consecutive_successes += 1
                self.consecutive_failures = 0
            else:
                self.consecutive_successes = 0
                self.consecutive_failures += 1
            
            # after evaluation window, check if we found better parameters
            if self.current_attempts >= self.evaluation_window:
                current_success_rate = self.current_successes / self.current_attempts
                
                # tracking history
                self.success_history.append(current_success_rate)
                self.parameter_history.append(self.current_params.copy())
                
                # if we found better parameters, update best
                if current_success_rate > self.best_success_rate:
                    improvement = current_success_rate - self.best_success_rate
                    self.best_success_rate = current_success_rate
                    self.best_params = self.current_params.copy()
                    self.episodes_without_improvement = 0
                    
                    print(f"Found better parameters! Success rate: {current_success_rate:.2%}")
                    print(f"Improvement: +{improvement:.2%}")
                    print(f"New best parameters: {self.best_params}")
                else:
                    self.episodes_without_improvement += 1
                    self._adjust_step_size()
                
                # try the new parameters
                self.current_params = self._try_new_parameters()
                self.current_successes = 0
                self.current_attempts = 0
    
    def get_action(self, obs):
        """Get action using current parameters"""
        self.stage_timer += 1
        
        # convert observations to float32
        obs = np.asarray(obs, dtype=np.float32)
        gripper_pos = obs[0:3]
        object_pos = obs[3:6].copy()
        object_pos[0] += np.float32(0.02)
        
        # calculate distances
        xy_dist = np.sqrt(
            np.float32(
                (object_pos[0] - gripper_pos[0])**2 + 
                (object_pos[1] - gripper_pos[1])**2
            )
        )
        
        # initialize action
        dx = np.float32(0.0)
        dy = np.float32(0.0)
        dz = np.float32(0.0)
        finger_angle = np.float32(1.0)
        
        if self.stage == 0:  # Approach
            dx = -np.float32(1.0) * (gripper_pos[0] - object_pos[0])
            dy = -np.float32(1.0) * (gripper_pos[1] - object_pos[1])
            
            target_height = object_pos[2] + self.current_params['approach_height']
            height_error = gripper_pos[2] - target_height
            dz = -np.float32(1.0) * height_error
            
            if not self.has_approached:
                if abs(height_error) < np.float32(0.01) and xy_dist < np.float32(0.01):
                    self.stage = 1
                    self.stage_timer = 0
                    self.has_approached = True
                
        elif self.stage == 1:  # Grasp
            dx = -self.current_params['grasp_correction'] * (gripper_pos[0] - object_pos[0])
            dy = -self.current_params['grasp_correction'] * (gripper_pos[1] - object_pos[1])
            dz = np.float32(-0.02)
            
            if self.stage_timer < 5:
                finger_angle = np.float32(1.0)
            else:
                finger_angle = np.float32(-1.0)
            
            if not self.has_grasped and self.stage_timer > 30:
                self.stage = 2
                self.stage_timer = 0
                self.has_grasped = True
            
        else:  # Lift
            dx = -self.current_params['lift_correction'] * (gripper_pos[0] - object_pos[0])
            dy = -self.current_params['lift_correction'] * (gripper_pos[1] - object_pos[1])
            dz = np.float32(0.15)
            finger_angle = np.float32(-1.0)
        
        # create action vector
        action = np.array([dx, dy, dz, finger_angle], dtype=np.float32)
        action[:3] *= self.action_scale
        
        # clip with float32 bounds
        action = np.clip(action, np.float32(-1.0), np.float32(1.0))
        
        # store last action
        self.last_action = action.copy()
        
        return action
    
    def save_policy(self, filename='src/hill_climbing_policy.pkl'):
        """Save policy parameters"""
        data = {
            'best_params': self.best_params,
            'best_success_rate': self.best_success_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_policy(self, filename='src/hill_climbing_policy.pkl'):
        """Load policy parameters"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.best_params = data['best_params']
            self.best_success_rate = data['best_success_rate']
            self.current_params = self.best_params.copy()