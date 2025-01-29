import numpy as np

class GraspingPolicy:
    def __init__(self):
        self.stage = 0  # 0: approach (align+descend), 1: grasp, 2: lift
        self.stage_timer = 0
    
    def reset(self):
        self.stage = 0
        self.stage_timer = 0
    
    def get_action(self, obs):
        # unpack observations
        gripper_pos = obs[0:3]
        object_pos = obs[3:6]
        object_pos[0] += 0.02 # slight offset on x position of object (for cube at least)
        self.stage_timer += 1
        
        # calculate distances
        xy_dist = np.sqrt((object_pos[0] - gripper_pos[0])**2 + 
                  (object_pos[1] - gripper_pos[1])**2)
        z_dist = gripper_pos[2] - object_pos[2]
        
        print(f"Stage: {self.stage}, XY Distance: {xy_dist:.3f}, Z Distance: {z_dist:.3f}")
        print(f"Gripper: {gripper_pos}")
        print(f"Object: {object_pos}")
        
        # Initialize action
        dx = dy = dz = 0.0
        finger_angle = 1.0  # start with open gripper
        
        # state machine for grasping
        if self.stage == 0:  # Approach (combined align and descend)
            # calculate target position above object
            dx = -1.0 * (gripper_pos[0] - object_pos[0])
            dy = -1.0 * (gripper_pos[1] - object_pos[1])
            
            # descend more slowly as we get closer
            target_height = object_pos[2] + 0.22  # target should be just above object
            height_error = gripper_pos[2] - target_height
            dz = -1.0 * height_error
            
            # transition when close to object and aligned
            if abs(height_error) < 0.01 and xy_dist < 0.01:
                print("Transitioning to grasp stage")
                self.stage = 1
                self.stage_timer = 0
                
        elif self.stage == 1:  # Grasp
            # maintain position with small corrections
            dx = -0.3 * (gripper_pos[0] - object_pos[0])
            dy = -0.3 * (gripper_pos[1] - object_pos[1])
            dz = -0.02  # a bit of gentle downward pressure
            
            if self.stage_timer < 5:  # brief pause before closing
                finger_angle = 1.0  # keep gripper open
            else:
                finger_angle = -1.0  # close gripper
                
            if self.stage_timer > 30:  # give kuka a bit of time for a proper grasp
                print("Transitioning to lift stage")
                self.stage = 2
                self.stage_timer = 0
                
        else:  # Lift
            # gentle corrections during lift
            dx = -0.2 * (gripper_pos[0] - object_pos[0])
            dy = -0.2 * (gripper_pos[1] - object_pos[1])
            dz = 0.15  # lift straight up
            finger_angle = -1.0  # keep gripper closed

        # create action vector
        action = np.array([dx, dy, dz, finger_angle])
        
        # scale down actions for stability
        action[:3] *= 0.15  # TODO: adjust this
        
        # clip actions for safety
        action = np.clip(action, -1, 1)
        
        print(f"Final action: {action}")
        return action