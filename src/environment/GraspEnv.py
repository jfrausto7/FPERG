from agent import kuka
import pybullet as p
import pybullet_data
import numpy as np
import random
import gym
from gym import spaces
import os

class GraspEnv(gym.Env):
    def __init__(self, gui=False):
        super().__init__()

        # init
        self.gui = gui
        self._urdfRoot = pybullet_data.getDataPath()
        self._timeStep = 1. / 120.  # controls speed of simulations in GUI (lower to 1/60 for super fast)
        self._maxSteps = 1000
        self._envStepCounter = 0
        self.terminated = 0
        
        # grasp success criteria
        self.initial_obj_height = -0.159
        self.lift_threshold = 0.05  # 5cm lift required
        self.lift_duration = 0      # Track how long object has been lifted
        self.required_lift_duration = 3.0  # Seconds
        
        # connect to physics server
        if gui:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # disable the GUI overlay
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # disable shadows
            p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # env setup
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        
        # action/observation spaces
        action_dim = 4  # dx, dy, dz, finger_angle
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        
        # obs space includes gripper and object states
        obs_dim = 6  # gripper_pos(3) + object_pos(3)
        obs_high = np.array([np.inf] * obs_dim)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        # init robot
        self.kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self.robot_id = self.kuka.kukaUid
        self.end_effector_index = 6

    def reset(self):
        self._envStepCounter = 0
        self.terminated = 0
        self.lift_duration = 0  # Reset lift duration
        
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        p.setGravity(0, 0, -10)
        
        # load plane and table
        p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])
        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
                  0.000000, 0.000000, 0.0, 1.0)
        
        # reset robot
        self.kuka = kuka.Kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self.robot_id = self.kuka.kukaUid
        
        # load objects with more randomness from a gaussian
        x = random.gauss(0.55, 0.05)  # Centered at 0.5, wider spread
        y = random.gauss(0.05, 0.05)  # Centered at 0.0, wider spread

        # List of possible objects
        object_files = (["cube_small.urdf"])
        #, "sphere_small.urdf", "teddy_vhacd.urdf"])

        # Can add any of these objects (performance decreases
        """object_files = (["cube_small.urdf", "sphere_small.urdf", "teddy_vhacd.urdf", "duck_vhacd.urdf"
        "/URDF_models/bowl/model.urdf", "/URDF_models/cleanser/model.urdf",
        "/URDF_models/blue_tea_box/model.urdf", "/URDF_models/blue_tea_box/model.urdf",
        "/URDF_models/medium_clamp/model.urdf", "/URDF_models/scissors/model.urdf"]"""

        selected_object = random.choice(object_files)
        self.object_id = p.loadURDF(os.path.join(self._urdfRoot, selected_object),
                                    [x, y, self.initial_obj_height],
                                    p.getQuaternionFromEuler([0, 0, 0]))

        # let objects settle w/ gravity
        for _ in range(20):
            p.stepSimulation()
            
        return self.get_observation()

    def get_observation(self):
        # get gripper state
        state = p.getLinkState(self.robot_id, self.end_effector_index)
        gripper_pos = state[0]
        
        # get object state
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        
        # combine observations (only positions)
        observation = np.array(list(gripper_pos) + list(obj_pos))
        return observation

    def step(self, action):
        self._envStepCounter += 1
        
        # scale actions
        dv = 0.05  # reduced velocity
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        finger_angle = 0.3 if action[3] > 0 else 0.0  # Binary gripper
        
        # apply action
        real_action = [dx, dy, dz, 0, finger_angle]
        self.kuka.applyAction(real_action)
        
        # step simulation
        p.stepSimulation()
        
        # get new state
        observation = self.get_observation()
        
        # check grasp success and update lift duration
        grasp_success = self.check_grasp_success()
        
        # compute reward
        reward = self._compute_reward()
        
        # check for termination state
        done = self._termination()
        
        info = {
            'grasp_success': grasp_success,
            'lift_duration': self.lift_duration,
            'steps': self._envStepCounter
        }
        
        return observation, reward, done, info

    def _compute_reward(self):
        """Compute reward based on lift height and gripper position"""
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        gripper_pos = p.getLinkState(self.robot_id, self.end_effector_index)[0]
        
        # Calculate distances
        lift_height = obj_pos[2] - self.initial_obj_height
        xy_dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(gripper_pos[:2]))
        
        reward = 0
        
        # Reward for getting closer to object
        if xy_dist < 0.1:  # Increased radius for positive feedback
            reward += (0.1 - xy_dist) * 10  # Scaled reward for proximity
        
        # Additional reward for correct height
        height_diff = abs(gripper_pos[2] - (obj_pos[2] + 0.1))  # Want gripper slightly above object
        if height_diff < 0.05:
            reward += (0.05 - height_diff) * 10
            
        # Major rewards for lifting
        if lift_height > self.lift_threshold:
            reward += 100
            if self.lift_duration >= self.required_lift_duration:
                reward += 1000
        
        return reward

    def check_grasp_success(self):
        """Check if grasp is successful based on lift height and duration"""
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        lift_height = obj_pos[2] - self.initial_obj_height
        is_lifted = lift_height > self.lift_threshold
        
        # update lift duration
        if is_lifted:
            self.lift_duration += self._timeStep
        else:
            self.lift_duration = 0  # reset if object drops
            
        # success only if lifted for required duration
        return self.lift_duration >= self.required_lift_duration

    def _termination(self):
        """Check if episode should end"""
        # end if max steps reached
        if self._envStepCounter > self._maxSteps:
            return True
            
        # end if grasp successful for required duration
        if self.check_grasp_success():
            return True
            
        # end if object fell off table
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        if obj_pos[2] < -0.3:
            return True
            
        return False

    def cleanup(self):
        p.disconnect()