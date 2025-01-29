import numpy as np
from lerobot.common.utils.gamepad_controller import ProController
import time
import mujoco
import json
import os
from scipy.spatial.transform import Rotation as R
from lerobot.common.utils.control_utils import SimulatedRobot, RealRobot, rotate_around_axis

def gamepad_control_robot(simulated_robot):
    """Interactive Pro Controller control for the robot with simulation preview"""
    # Initialize controller
    controller = ProController()
    if not controller.connect():
        print("Failed to initialize controller")
        return
    
    # Initialize robot using RealRobot class with higher update frequency
    robot = RealRobot()
    robot.update_frequency = 200  # Increase motion control thread frequency
    
    # Read initial position from robot
    initial_angles_sim = robot.read(to_sim=True)
    print("Initial robot angles:", initial_angles_sim)
    
    # Calculate initial position using forward kinematics
    simulated_robot.data.qpos[simulated_robot.actuated_joint_ids] = initial_angles_sim
    mujoco.mj_forward(simulated_robot.model, simulated_robot.data)
    
    end_effector_id = 5
    current_pos = simulated_robot.data.xpos[end_effector_id].copy()
    current_orientation = simulated_robot.data.xmat[end_effector_id].reshape(3, 3).copy()
    initial_pos = current_pos.copy()
    initial_orientation = current_orientation.copy()
    
    print("Initial position:", current_pos)
    
    # Define workspace bounds
    xyz_min = np.array([-0.4, -0.4, 0.0])
    xyz_max = np.array([0.4, 0.1, 0.4])
    
    # Initialize previous IK solution
    previous_ik = initial_angles_sim
    
    # Increased step sizes for faster response
    pos_step_size = 0.001
    rot_step_size = 0.015   
    
    # Improved deadzone thresholds
    stick_deadzone = 0.15  # Increased deadzone for sticks
    rot_deadzone = 1.0   # Separate deadzone for rotation, 1 to deactivate
    
    # Track positions
    target_pos = current_pos.copy()
    target_orientation = current_orientation.copy()
    last_successful_pos = current_pos.copy()
    last_successful_angles = previous_ik.copy()
    last_command_time = time.time()
    movement_active = False
    fine_control_active = False
    plus_pressed_last_frame = False
    minus_pressed_last_frame = False
    
    # Gripper control parameters
    GRIPPER_OPEN_ANGLE = np.radians(80)
    GRIPPER_CLOSED_ANGLE = np.radians(-13)
    current_gripper_angle = GRIPPER_CLOSED_ANGLE
    gripper_step = np.radians(0.5)  # Speed of gripper movement
    
    # Store home position
    home_pos = initial_pos.copy()
    home_orientation = initial_orientation.copy()
    home_angles = initial_angles_sim.copy()
    
    # Add parameters for position commands near start of function
    COMMAND_DURATION = 200  # Number of steps to keep sending position commands
    home_counter = 0
    stored_pos_counter = 0
    
    # Add storage for memorized positions
    stored_positions = {
        'Up': {'pos': None, 'orientation': None, 'angles': None},
        'Down': {'pos': None, 'orientation': None, 'angles': None},
        'Left': {'pos': None, 'orientation': None, 'angles': None},
        'Right': {'pos': None, 'orientation': None, 'angles': None}
    }
    store_mode = False
    
    print("\nControls:")
    print("Left Stick: Move in X-Y plane")
    print("Right Stick Y-axis: Move in Z axis")
    print("Right Stick X-axis: Rotate gripper")
    print("L Button: Open gripper progressively")
    print("R Button: Close gripper progressively")
    print("Minus Button: Toggle fine control")
    print("Home Button: Return to home position")
    print("Capture Button: Exit")
    print("Plus: Toggle store mode")
    print("D-pad in store mode: Store current position")
    print("D-pad: Go to stored position")

    try:
        viewer = mujoco.viewer.launch_passive(simulated_robot.model, simulated_robot.data)
        last_update_time = time.time()
        
        while viewer.is_running():
            current_time = time.time()
            
            # Increased polling rate with precise timing
            elapsed = current_time - last_update_time
            if elapsed < 0.001:  # 1000Hz polling
                continue
                
            last_update_time = current_time
            state = controller.read_state()
            
            if state:
                # Check for exit condition
                if 'Capture' in state['buttons']:
                    break
                    
                # Check for home position command
                if 'Home' in state['buttons']:
                    home_counter = COMMAND_DURATION
                
                # Keep sending home position for several steps
                if home_counter > 0:
                    target_pos = home_pos.copy()
                    target_orientation = home_orientation.copy()
                    robot.write(home_angles, from_sim=True)
                    joint_angles = robot.read(to_sim=True)
                    simulated_robot.data.qpos[simulated_robot.actuated_joint_ids] = joint_angles
                    mujoco.mj_forward(simulated_robot.model, simulated_robot.data)
                    last_successful_angles = joint_angles.copy()
                    last_successful_pos = target_pos.copy()
                    last_command_time = current_time
                    movement_active = True
                    home_counter -= 1
                    continue  # Skip regular movement processing while homing
                
                # Handle Plus button toggle
                if 'Plus' in state['buttons'] and not plus_pressed_last_frame:
                    store_mode = not store_mode
                    print("Store mode:", "ON" if store_mode else "OFF")
                plus_pressed_last_frame = 'Plus' in state['buttons']

                # Handle position storage and recall
                if store_mode:
                    # Store positions when in store mode and direction is pressed
                    for direction in ['Up', 'Down', 'Left', 'Right']:
                        if direction in state['buttons']:
                            current_angles = robot.read(to_sim=True)
                            stored_positions[direction] = {
                                'pos': last_successful_pos.copy(),
                                'orientation': target_orientation.copy(),
                                'angles': current_angles.copy()
                            }
                            print(f"Position stored for {direction}")
                            store_mode = False  # Exit store mode after storing
                            print("Store mode: OFF")
                else:
                    # Recall positions when not in store mode
                    for direction in ['Up', 'Down', 'Left', 'Right']:
                        if direction in state['buttons'] and stored_positions[direction]['angles'] is not None:
                            # Get stored values
                            stored_angles = stored_positions[direction]['angles'].copy()
                            
                            # Set target directly to stored joint angles
                            target_pos = stored_positions[direction]['pos'].copy()
                            target_orientation = stored_positions[direction]['orientation'].copy()
                            
                            # Update control system targets
                            last_successful_angles = stored_angles.copy()
                            last_successful_pos = target_pos.copy()
                            
                            # Write desired joint angles directly once
                            robot.write(stored_angles, from_sim=True)
                            
                            # Update simulation state
                            simulated_robot.data.qpos[simulated_robot.actuated_joint_ids] = stored_angles
                            mujoco.mj_forward(simulated_robot.model, simulated_robot.data)
                            
                            # Reset command timer to keep movement active
                            last_command_time = current_time
                
                # Handle button toggles
                if 'Minus' in state['buttons'] and not minus_pressed_last_frame:
                    fine_control_active = not fine_control_active
                    print(f"Fine control: {'ON' if fine_control_active else 'OFF'}")
                minus_pressed_last_frame = 'Minus' in state['buttons']

                # Calculate fine control mode based on toggle
                fine_control = 0.2 if fine_control_active else 1.0
                current_pos_step = pos_step_size * fine_control
                current_rot_step = rot_step_size * fine_control
                current_gripper_step = gripper_step * fine_control
                
                # Get stick values with deadzone
                left_stick = state['sticks']['left']
                right_stick = state['sticks']['right']
                
                # Apply deadzone
                lx = 0.0 if abs(left_stick['x']) < stick_deadzone else left_stick['x']
                ly = 0.0 if abs(left_stick['y']) < stick_deadzone else left_stick['y']
                rx = 0.0 if abs(right_stick['x']) < rot_deadzone else right_stick['x']
                ry = 0.0 if abs(right_stick['y']) < stick_deadzone else right_stick['y']
                
                # Check if any movement is active
                movement_active = any([abs(v) > 0 for v in [lx, ly, rx, ry]])
                
                # Progressive gripper control with R/L buttons
                gripper_changed = False
                if 'L' in state['buttons']:
                    # Open gripper progressively
                    new_angle = current_gripper_angle + current_gripper_step
                    if new_angle <= GRIPPER_OPEN_ANGLE:
                        current_gripper_angle = new_angle
                        gripper_changed = True
                elif 'R' in state['buttons']:
                    # Close gripper progressively
                    new_angle = current_gripper_angle - current_gripper_step
                    if new_angle >= GRIPPER_CLOSED_ANGLE:
                        current_gripper_angle = new_angle
                        gripper_changed = True
                
                # Update target position only if movement is active
                if movement_active:
                    target_pos[0] -= lx * current_pos_step
                    target_pos[1] -= ly * current_pos_step
                    target_pos[2] += ry * current_pos_step
                    last_command_time = current_time
                    
                    # Rotate end effector
                    if abs(rx) > 0:
                        target_orientation = rotate_around_axis(
                            target_orientation, 
                            'z', 
                            rx * current_rot_step
                        )

                # Clip target position to workspace bounds
                target_pos = np.clip(target_pos, xyz_min, xyz_max)
                
                # Update robot if there's movement, recent command, or gripper change
                if movement_active or (current_time - last_command_time) < 0.1 or gripper_changed:
                    # Try IK with increased iterations and optimized parameters
                    joint_angles = simulated_robot.inverse_kinematics(
                        target_pos,
                        target_orientation,
                        current_angles=last_successful_angles,
                        max_iter=30,  # Reduced from 50 for faster response
                        orientation_weight=0.001  # Reduced from 0.01 for better orientation tracking
                    )
                    
                    if joint_angles is not None:
                        # Update successful position and angles
                        last_successful_pos = target_pos.copy()
                        last_successful_angles = joint_angles.copy()
                        
                        # Set current gripper position
                        joint_angles[-1] = current_gripper_angle
                        
                        # Update simulation and robot
                        robot.write(joint_angles, from_sim=True)
                        joint_angles = robot.read(to_sim=True)
                        joint_angles[-1] = current_gripper_angle
                        simulated_robot.data.qpos[simulated_robot.actuated_joint_ids] = joint_angles
                        mujoco.mj_forward(simulated_robot.model, simulated_robot.data)
                    else:
                        # If IK fails, stay at last successful position
                        target_pos = last_successful_pos.copy()
                        print("\nIK failed - position reset")
                
            viewer.sync()

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        controller.disconnect()
        del robot
        if 'viewer' in locals():
            viewer.close()

if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = repo_dir + "/configs/robot/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf"
    simulated_robot = SimulatedRobot(urdf_path)
    
    gamepad_control_robot(simulated_robot)