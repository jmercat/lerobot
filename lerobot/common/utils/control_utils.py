import numpy as np
import json
import time
import threading
from typing import Optional
import mujoco
import mujoco.viewer
from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

def rotate_around_axis(orientation_matrix, axis, angle):
    """
    Rotate an orientation matrix around a specified axis by a given angle.
    
    Args:
        orientation_matrix (np.ndarray): 3x3 rotation matrix
        axis (str): Axis of rotation ('x', 'y', or 'z')
        angle (float): Angle of rotation in radians
    
    Returns:
        np.ndarray: New 3x3 rotation matrix after applying the rotation
    """
    # Create rotation matrix for the specified axis
    c = np.cos(angle)
    s = np.sin(angle)
    
    if axis.lower() == 'x':
        rotation = np.array([
            [1, 0,  0],
            [0, c, -s],
            [0, s,  c]
        ])
    elif axis.lower() == 'y':
        rotation = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c]
        ])
    elif axis.lower() == 'z':
        rotation = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
    
    # Apply rotation to the current orientation
    return rotation @ orientation_matrix

class SimulatedRobot:
    def __init__(self, urdf_path) -> None:
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)
        
        # Find joint indices
        self.actuated_joint_ids = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] == 3:  # 3 = hinge joint (rotational)
                self.actuated_joint_ids.append(i)
        
        print(f"Found {len(self.actuated_joint_ids)} actuated joints")
        print(f"Joint types: {[self.model.jnt_type[i] for i in range(self.model.njnt)]}")
        print(f"Joint names: {[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(self.model.njnt)]}")

        # Define joint limits (in radians)
        self.joint_limits = {
            'lower': np.array([-83, -204, -198, -184, -180, -15]) * np.pi/180,
            'upper': np.array([130, 4, 1, 31, 340, 80]) * np.pi/180
        }

        # Joint offsets and directions
        self.joint_offsets = np.array([
            0.0,            # joint_0 (base)
            np.pi/2,       # joint_1 (shoulder)
            -np.pi/2,      # joint_2 (elbow)
            -np.pi/2,      # joint_3 (wrist flex)
            np.pi,         # joint_4 (wrist roll)
        ])
        
        self.joint_directions = np.array([
            1.0,    # joint_0
            1.0,    # joint_1
            -1.0,   # joint_2
            1.0,    # joint_3
            1.0,    # joint_4
        ])

        # Static gripper body is the 5th body in the model
        # 0: base
        # 1: shoulder
        # 2: lower arm
        # 3: upper arm
        # 4: wrist
        # 5: gripper fixed
        # 6: gripper moving
        self.end_effector_id = 5

        print(f"Using end effector body: {mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.end_effector_id)}")

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector position and orientation using MuJoCo"""
        self.data.qpos[self.actuated_joint_ids] = joint_angles
        mujoco.mj_forward(self.model, self.data)
        
        position = self.data.xpos[self.end_effector_id].copy()
        rotation = self.data.xmat[self.end_effector_id].reshape(3, 3).copy()
        
        return position, rotation

    def inverse_kinematics(self, target_pos, target_orientation=None, current_angles=None, max_iter=100, orientation_weight=0.5):
        """Calculate joint angles using MuJoCo's Jacobian"""
        # Only use first 5 joints (excluding gripper and gripper rotation)
        actuated_joint_ids = self.actuated_joint_ids  # Changed from [:-1] to [:5]
        non_actuated_joint_ids = []
        rot_error = 0.0
        
        if current_angles is not None:
            self.data.qpos[actuated_joint_ids] = current_angles[actuated_joint_ids]
        
        # Adjust parameters for better convergence
        alpha = 0.1  # Reduced step size further for more stability
        tolerance = 1e-2  # Slightly increased tolerance
        min_improvement = 1e-6  # Minimum improvement threshold
        last_error = float('inf')
        stall_count = 0
        max_stall = 10  # Maximum number of iterations without improvement

        for iteration in range(max_iter):
            # Get current end effector position and orientation
            mujoco.mj_forward(self.model, self.data)
            current_pos = self.data.xpos[self.end_effector_id].copy()
            
            # Calculate position error
            pos_error = target_pos - current_pos
            
            # Initialize Jacobian
            jac = np.zeros((6, self.model.nv))
            
            # Get Jacobian for position and orientation
            mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], self.data.xpos[self.end_effector_id], self.end_effector_id)
            
            # Use only the columns corresponding to actuated joints
            jac = jac[:, actuated_joint_ids]
            
            if target_orientation is not None:
                current_rot = self.data.xmat[self.end_effector_id].reshape(3, 3)
                # Improved rotation error calculation using logarithm of rotation matrix
                R_error = target_orientation @ current_rot.T
                angle = np.arccos((np.trace(R_error) - 1) / 2)
                if angle > 1e-10:
                    rot_error = angle / (2 * np.sin(angle)) * np.array([
                        R_error[2,1] - R_error[1,2],
                        R_error[0,2] - R_error[2,0],
                        R_error[1,0] - R_error[0,1]
                    ])
                else:
                    rot_error = np.zeros(3)
                
                total_error = np.concatenate([pos_error, rot_error * orientation_weight])  # Scale rotation error
                # Use full Jacobian
            else:
                # Use only position Jacobian
                jac = jac[:3]
                total_error = pos_error
            
            # Check convergence
            error_norm = np.linalg.norm(total_error)
            
            # Check if we're making progress
            if abs(last_error - error_norm) < min_improvement:
                stall_count += 1
            else:
                stall_count = 0
            
            if error_norm < tolerance:
                print(f"IK converged after {iteration} iterations, error: {error_norm}")
                break
            
            if stall_count >= max_stall:
                print(f"IK stalled after {iteration} iterations, error: {error_norm}")
                # Return best solution so far
                break
            
            # Adaptive damping based on error magnitude
            lambda_ = 0.1 * (1 + 100 * error_norm)  # Increased damping factor
            
            jac_t = jac.T
            J_JT = jac @ jac_t
            delta_q = jac_t @ np.linalg.solve(J_JT + lambda_ * np.eye(J_JT.shape[0]), total_error)
            
            # More conservative step size limit
            max_step = 0.05  # Further reduced maximum step size
            step_norm = np.linalg.norm(delta_q)
            if step_norm > max_step:
                delta_q = delta_q * max_step / step_norm
            
            # Update joint positions with momentum
            momentum = 0.5  # Add momentum term
            if iteration > 0:
                delta_q = momentum * previous_delta_q + (1 - momentum) * delta_q
            previous_delta_q = delta_q.copy()
            
            self.data.qpos[actuated_joint_ids] += alpha * delta_q
            if current_angles is not None:
                self.data.qpos[non_actuated_joint_ids] = current_angles[non_actuated_joint_ids]
            
            # Apply joint limits with smooth transition
            joint_positions = self.data.qpos[actuated_joint_ids]
            for i in range(len(joint_positions)):
                if joint_positions[i] < self.joint_limits['lower'][i]:
                    joint_positions[i] = self.joint_limits['lower'][i]
                elif joint_positions[i] > self.joint_limits['upper'][i]:
                    joint_positions[i] = self.joint_limits['upper'][i]
            
            if iteration == max_iter - 1:
                print(f"IK failed to converge after {max_iter} iterations, final error: {error_norm}")
                print(f"    Position error: {pos_error}")
                print(f"    Rotation error: {rot_error}")
        
            last_error = error_norm

        return self.data.qpos.copy()

    def mujoco_to_sim_angles(self, mujoco_angles):
        """Convert MuJoCo angles to simulation angles"""
        return (mujoco_angles * self.joint_directions) + self.joint_offsets

    def sim_to_mujoco_angles(self, sim_angles):
        """Convert simulation angles to MuJoCo angles"""
        return (sim_angles - self.joint_offsets) * self.joint_directions
    

class SmoothMotionController:
    def __init__(self, max_speed: float = 300.0, acceleration: float = 300.0):
        """
        Args:
            max_speed: Maximum speed in degrees per second
            acceleration: Acceleration in degrees per second²
        """
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.current_pos = 0.0
        self.current_velocity = 0.0
        self.target_pos = 0.0
        self.last_update_time = time.time()

    def update(self, current_time: float) -> float:
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Calculate distance to target
        distance = self.target_pos - self.current_pos
        
        # Calculate stopping distance at current velocity
        stopping_dist = (self.current_velocity ** 2) / (2 * self.acceleration)
        
        # Determine if we need to accelerate or decelerate
        if abs(distance) > stopping_dist:
            # Accelerate
            desired_velocity = np.sign(distance) * self.max_speed
        else:
            # Decelerate
            desired_velocity = 0.0
        
        # Apply acceleration limits with higher responsiveness
        velocity_change = desired_velocity - self.current_velocity
        max_change = self.acceleration * dt
        velocity_change = np.clip(velocity_change, -max_change, max_change)
        self.current_velocity += velocity_change
        
        # Update position
        self.current_pos += self.current_velocity * dt
        
        return self.current_pos

class RealRobot:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RealRobot, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if RealRobot._initialized:
            return
            
        # Robot setup
        self.follower_port = "/dev/tty.usbmodem585A0084011"
        self.arm_calib_path = ".cache/calibration/so100"
        
        # Add retry logic for connection
        max_retries = 3
        retry_delay = 1.0  # seconds
        
        self.arm = FeetechMotorsBus(
            port=self.follower_port,
            motors={
                "shoulder_pan": (1, "sts3215"),
                "shoulder_lift": (2, "sts3215"),
                "elbow_flex": (3, "sts3215"),
                "wrist_flex": (4, "sts3215"),
                "wrist_roll": (5, "sts3215"),
                "gripper": (6, "sts3215"),
            },
        )

        with open(self.arm_calib_path + "/main_follower.json") as f:
            follower_calibration = json.load(f)

        self.arm.connect()
        self.arm.set_calibration(follower_calibration)

        # Rest of initialization code...
        self.angle_min = np.array([-85, -12, -18, -104, -270, -15])
        self.angle_max = np.array([127, 195, 180, 110, 270, 80])
        self.home_pos = np.array([2, 191.25, 179.29688, 79.541016, -87.802734, 13])
        self.sim_zero = np.array([0, -103.13240312, -90.52733163, 15.46986047, 90, 0])

        # Motion parameters
        self.update_frequency = 100  # Hz
        self.position_tolerance = 0.1  # degrees
        
        # Motion control setup with momentum
        self.controllers = [SmoothMotionController() for _ in range(6)]
        self._lock = threading.Lock()
        self.current_angles = self.read()
        self.previous_positions = self.current_angles.copy()
        for i, ctrl in enumerate(self.controllers):
            ctrl.current_pos = self.current_angles[i]
            ctrl.target_pos = self.current_angles[i]
        
        # Threading setup
        self._running = False
        self._motion_thread = None
        
        # Start motion control thread
        self.start()
        
        RealRobot._initialized = True

    def __del__(self):
        """Cleanup when the object is destroyed"""
        if RealRobot._initialized:
            try:
                self.stop()
                if hasattr(self, 'arm'):
                    self.arm.disconnect()
                RealRobot._initialized = False
                RealRobot._instance = None
            except Exception as e:
                print(f"Error during cleanup: {e}")

    def start(self):
        """Start the motion control thread"""
        if not self._running:
            self._running = True
            self._motion_thread = threading.Thread(target=self._motion_control_loop)
            self._motion_thread.daemon = True
            self._motion_thread.start()

    def stop(self):
        """Stop the motion control thread"""
        self._running = False
        if self._motion_thread:
            self._motion_thread.join()
            self._motion_thread = None

    def _motion_control_loop(self):
        """Main motion control loop"""
        while self._running:
            current_time = time.time()
            
            with self._lock:
                # Update all joint positions
                new_positions = np.array([
                    ctrl.update(current_time) for ctrl in self.controllers
                ])
                
                # Clip to joint limits
                new_positions = np.clip(new_positions, self.angle_min, self.angle_max)
                
                # Apply momentum for smoother motion
                momentum = 0.5  # Adjust this value between 0 and 1
                smoothed_positions = momentum * self.previous_positions + (1 - momentum) * new_positions
                self.previous_positions = smoothed_positions.copy()
                
                try:
                    # Write to hardware using Goal_Position instead of Present_Position
                    self.arm.write("Goal_Position", smoothed_positions)
                except ConnectionError as e:
                    print(f"Communication error: {e}")
                    time.sleep(0.01)
                    continue
            
            # Sleep to maintain update frequency
            time.sleep(1.0 / self.update_frequency)

    def read(self, to_sim=False):
        with self._lock:
            angles = self.arm.read("Present_Position")
        if to_sim:
            angles = np.radians(angles - self.home_pos - self.sim_zero)
            angles[0] = -angles[0]
            angles[1] = -angles[1]
            angles[4] = -angles[4]
        return angles

    def write(self, angles, from_sim=False):
        """Set new target positions for the joints"""
        if from_sim:
            angles[0] = -angles[0]
            angles[1] = -angles[1]
            angles[4] = -angles[4]
            angles = np.degrees(angles) + self.home_pos + self.sim_zero
        
        # Clip angles to joint limits
        angles = np.clip(angles, self.angle_min, self.angle_max)
        
        # Update target positions in controllers
        with self._lock:
            for i, ctrl in enumerate(self.controllers):
                ctrl.target_pos = angles[i]

    def is_moving(self) -> bool:
        """Check if any joint is still moving"""
        with self._lock:
            return any(
                abs(ctrl.current_pos - ctrl.target_pos) > self.position_tolerance
                or abs(ctrl.current_velocity) > 0.1
                for ctrl in self.controllers
            )

    def wait_until_stopped(self, timeout: Optional[float] = None):
        """Wait until all joints have reached their targets"""
        start_time = time.time()
        while self.is_moving():
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)
        return True
