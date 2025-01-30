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
    # Add pick/place variables
    stored_pick_position = None
    stored_pick_height = None
    a_pressed_last_frame = False
    b_pressed_last_frame = False
    
    # Update position tolerance for sequence steps
    POSITION_TOLERANCE = 0.005  # 5mm tolerance
    GRIPPER_TOLERANCE = 0.05    # Tolerance for gripper angle
    POSITION_TIMEOUT = 2.0  # Timeout for position moves
    
    # Gripper control parameters
    GRIPPER_OPEN_ANGLE = np.radians(30)
    GRIPPER_CLOSED_ANGLE = np.radians(-13)
    current_gripper_angle = GRIPPER_CLOSED_ANGLE
    target_gripper_angle = GRIPPER_CLOSED_ANGLE
    previous_gripper_angle = GRIPPER_CLOSED_ANGLE  # Initialize previous angle
    gripper_step = np.radians(0.5)
    grip_start_time = None  # Track when grip attempt started

    # Add pick/place state variables
    pick_sequence_active = False
    place_sequence_active = False
    pick_place_height = 0.14
    above_table_height = 0.2
    pick_state = 0  # 0: start, 1: opening, 2: wait_open, 3: moving down, 4: closing, 5: wait_close, 6: moving up
    place_state = 0  # 0: start, 1: moving down, 2: wait_down, 3: opening, 4: wait_open, 5: moving up, 6: closing, 7: wait_close
    sequence_target_reached = True  # Flag to track if current sequence target is reached
    
    # Store home position
    home_pos = initial_pos.copy()
    home_orientation = initial_orientation.copy()
    home_angles = initial_angles_sim.copy()
    
    # Add home to stored positions
    stored_positions = {
        'Home': {'pos': home_pos, 'orientation': home_orientation, 'angles': home_angles},
        'Up': {'pos': None, 'orientation': None, 'angles': None},
        'Down': {'pos': None, 'orientation': None, 'angles': None},
        'Left': {'pos': None, 'orientation': None, 'angles': None},
        'Right': {'pos': None, 'orientation': None, 'angles': None}
    }
    store_mode = False
    
    # Add IK parameters for different movement types
    MANUAL_IK_PARAMS = {
        'max_iter': 50,
        'orientation_weight': 0.005,
        'tolerance': 0.05
    }

    FINE_IK_PARAMS = {
        'max_iter': 100,
        'orientation_weight': 0.05,
        'tolerance': 0.02
    }
    
    SEQUENCE_IK_PARAMS = {
        'max_iter': 200,
        'orientation_weight': 0.01,
        'tolerance': 0.01
    }

    def try_ik_solution(target_pos, target_orientation, last_successful_angles, is_sequence=False, is_fine=False):
        """Helper function to compute IK with appropriate parameters"""
        params = SEQUENCE_IK_PARAMS if is_sequence else MANUAL_IK_PARAMS if is_fine else FINE_IK_PARAMS
        return simulated_robot.inverse_kinematics(
            target_pos,
            target_orientation,
            current_angles=last_successful_angles,
            **params
        )

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
    print("A Button: Start/cancel pick sequence (open gripper, move down, close, return up)")
    print("B Button: Start/cancel place sequence (move down, open gripper, return up)")

    try:
        viewer = mujoco.viewer.launch_passive(simulated_robot.model, simulated_robot.data)
        last_update_time = time.time()
        
        while viewer.is_running():
            current_time = time.time()
            current_angles = robot.read(to_sim=True)
            current_gripper_angle = current_angles[-1]
            current_pos, current_orientation = simulated_robot.forward_kinematics(current_angles)
            
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
                movement_active = any([abs(v) > 0 for v in [lx, ly, rx, ry]]) or abs(current_gripper_angle - target_gripper_angle) > GRIPPER_TOLERANCE
                
                # Progressive gripper control with R/L buttons
                if 'L' in state['buttons']:
                    # Open gripper progressively
                    new_angle = target_gripper_angle + current_gripper_step
                    if new_angle <= GRIPPER_OPEN_ANGLE:
                        target_gripper_angle = new_angle
                elif 'R' in state['buttons']:
                    # Close gripper progressively
                    new_angle = target_gripper_angle - current_gripper_step
                    if new_angle >= GRIPPER_CLOSED_ANGLE:
                        target_gripper_angle = new_angle
                
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
                
                # Update robot if there's movement, recent command, gripper change, or sequence is active
                if movement_active or (current_time - last_command_time) < 0.1:
                    # Try IK with appropriate parameters
                    is_sequence = pick_sequence_active or place_sequence_active
                    joint_angles = try_ik_solution(target_pos, target_orientation, last_successful_angles, is_sequence, is_fine=fine_control_active)
                    
                    if joint_angles is not None:
                        # Update successful position and angles
                        last_successful_pos = target_pos.copy()
                        last_successful_angles = joint_angles.copy()
                        
                        # Set target gripper position
                        joint_angles[-1] = target_gripper_angle

                        # Update simulation and robot
                        robot.write(joint_angles, from_sim=True)
                    else:
                        # If IK fails, stay at last successful position
                        target_pos = last_successful_pos.copy()
                        print("\nIK failed - position reset")
                
                # Handle home button
                if 'Home' in state['buttons'] and stored_positions['Home']['angles'] is not None:
                    # Get stored home values
                    stored_angles = stored_positions['Home']['angles'].copy()
                    
                    # Set target directly to stored joint angles
                    target_pos = stored_positions['Home']['pos'].copy()
                    target_orientation = stored_positions['Home']['orientation'].copy()
                    
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

                # Handle A button for pick sequence
                if 'A' in state['buttons'] and not a_pressed_last_frame:
                    if not pick_sequence_active:
                        # Start pick sequence
                        pick_sequence_active = True
                        pick_state = 0
                        stored_pick_position = target_pos.copy()
                        stored_pick_height = target_pos[2]
                        sequence_target_reached = True
                        print("Starting pick sequence")
                    else:
                        # Cancel sequence
                        pick_sequence_active = False
                        pick_state = 0
                        print("Pick sequence cancelled")
                a_pressed_last_frame = 'A' in state['buttons']

                # Handle B button for place sequence
                if 'B' in state['buttons'] and not b_pressed_last_frame:
                    if not place_sequence_active and stored_pick_height is not None:
                        # Start place sequence
                        place_sequence_active = True
                        place_state = 0
                        sequence_target_reached = True
                        print("Starting place sequence")
                    else:
                        # Cancel sequence
                        place_sequence_active = False
                        place_state = 0
                        print("Place sequence cancelled")
                b_pressed_last_frame = 'B' in state['buttons']

                # Handle pick sequence state machine
                if pick_sequence_active:
                    place_sequence_active = False
                    place_state = 0
                    
                    # Get current gripper angle from robot
                    current_gripper_angle = robot.read(to_sim=True)[-1]
                    
                    # Check if we've reached the current target with tight tolerances
                    position_error = np.linalg.norm(current_pos - target_pos)
                    position_reached = position_error < POSITION_TOLERANCE
                    
                    # Different gripper checks for opening vs closing
                    if target_gripper_angle == GRIPPER_OPEN_ANGLE:
                        # When opening, we want to reach the full angle
                        gripper_reached = abs(current_gripper_angle - target_gripper_angle) < GRIPPER_TOLERANCE
                    else:
                        # When closing, consider it reached if:
                        # 1. We reached target angle (normal close) OR
                        # 2. Gripper stopped moving (gripped object) OR
                        # 3. Timeout reached (prevent getting stuck)
                        gripper_stopped = abs(current_gripper_angle - previous_gripper_angle) < 0.001
                        if grip_start_time is None and pick_state == 4:
                            grip_start_time = current_time
                        
                        timeout_reached = (grip_start_time is not None and 
                                         current_time - grip_start_time > POSITION_TIMEOUT)
                        
                        gripper_reached = (abs(current_gripper_angle - target_gripper_angle) < GRIPPER_TOLERANCE or 
                                         gripper_stopped or timeout_reached)
                    
                    if pick_state == 0:  # Start opening gripper
                        print("Opening gripper...")
                        grip_start_time = None
                        target_gripper_angle = GRIPPER_OPEN_ANGLE
                        # The gripper oppens to the left side so a small offset is added to the x position
                        target_pos[0] -= 0.01
                        pick_state = 1
                    elif pick_state == 1:  # Wait for gripper to open
                        target_gripper_angle = GRIPPER_OPEN_ANGLE
                        if gripper_reached:
                            print("Gripper open complete")
                            pick_state = 2
                    elif pick_state == 2:  # Start moving down
                        print("Moving down...")
                        # Remove the offset added to the x position
                        target_pos[0] += 0.01
                        pick_state = 3
                    elif pick_state == 3:  # Moving down
                        target_pos[2] = pick_place_height
                        # Start timeout tracking if not started
                        if grip_start_time is None:
                            grip_start_time = current_time
                            
                        # Check if position reached or timeout
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        
                        if position_reached or timeout_reached:
                            if timeout_reached:
                                print(f"Position timeout reached (error: {position_error:.4f}m)")
                            else:
                                print(f"Down position reached (error: {position_error:.4f}m)")
                            
                            print("Starting gripper close...")
                            grip_start_time = current_time
                            target_gripper_angle = GRIPPER_CLOSED_ANGLE
                            pick_state = 4
                    elif pick_state == 4:  # Closing gripper
                        previous_gripper_angle = current_gripper_angle
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if gripper_reached:
                            print("Grip complete")
                            grip_start_time = current_time
                            pick_state = 5
                    elif pick_state == 5:  # Wait after closing
                        # Additional wait state to ensure grip is stable
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if abs(current_gripper_angle - target_gripper_angle) < GRIPPER_TOLERANCE or timeout_reached:
                            print("Starting move up...")
                            move_up_start_time = current_time
                            pick_state = 6
                    elif pick_state == 6:  # Moving up
                        target_pos[2] = stored_pick_height
                        timeout_reached = (current_time - move_up_start_time) > POSITION_TIMEOUT/2
                        if position_reached or timeout_reached:
                            print("Pick sequence complete")
                            pick_sequence_active = False
                            pick_state = 0

                # Handle place sequence state machine
                if place_sequence_active:
                    pick_sequence_active = False
                    pick_state = 0
                    
                    # Get current gripper angle from robot
                    current_gripper_angle = robot.read(to_sim=True)[-1]
                    
                    # Check if we've reached the current target with tight tolerances
                    position_error = np.linalg.norm(current_pos - target_pos)
                    position_reached = position_error < POSITION_TOLERANCE
                    gripper_reached = abs(current_gripper_angle - target_gripper_angle) < GRIPPER_TOLERANCE
                    
                    # Store current position before attempting moves
                    if place_state in [0, 1, 4, 5]:  # States involving position changes
                        last_successful_pos = current_pos.copy()
                        last_successful_angles = robot.read(to_sim=True)
                    
                    if place_state == 0:  # Start moving down
                        print("Moving down...")
                        target_pos = current_pos.copy()  # Start from current position
                        grip_start_time = current_time
                        place_state = 1
                    elif place_state == 1:  # Moving down
                        target_pos[2] = pick_place_height
                        # Try IK solution
                        joint_angles = try_ik_solution(target_pos, target_orientation, last_successful_angles, True)
                        
                        if joint_angles is None:
                            print("IK failed during down movement - resetting position")
                            target_pos = last_successful_pos.copy()
                            continue
                            
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        
                        if position_reached or timeout_reached:
                            if timeout_reached:
                                print(f"Position timeout reached (error: {position_error:.4f}m)")
                            else:
                                print(f"Down position reached (error: {position_error:.4f}m)")
                            
                            grip_start_time = current_time
                            place_state = 2
                    elif place_state == 2:  # Wait after moving down
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if timeout_reached:
                            print("Starting gripper open...")
                            grip_start_time = current_time
                            target_gripper_angle = GRIPPER_OPEN_ANGLE
                            place_state = 3
                    elif place_state == 3:  # Opening gripper
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if gripper_reached or timeout_reached:
                            print("Gripper open complete")
                            grip_start_time = current_time
                            place_state = 4
                    elif place_state == 4:  # Wait after opening
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if timeout_reached:
                            print("Starting move up...")
                            target_pos = current_pos.copy()  # Start from current position
                            move_up_start_time = current_time
                            place_state = 5
                    elif place_state == 5:  # Moving up
                        target_pos[2] = max(stored_pick_height, above_table_height)
                        # Try IK solution
                        joint_angles = try_ik_solution(target_pos, target_orientation, last_successful_angles, True)
                        
                        if joint_angles is None:
                            print("IK failed during up movement - resetting position")
                            target_pos = last_successful_pos.copy()
                            continue
                            
                        timeout_reached = (current_time - move_up_start_time) > POSITION_TIMEOUT
                        if position_reached or timeout_reached:
                            print("Move up complete, closing gripper...")
                            grip_start_time = current_time
                            target_gripper_angle = GRIPPER_CLOSED_ANGLE
                            place_state = 6
                    elif place_state == 6:  # Closing gripper
                        if gripper_reached:
                            print("Gripper closed")
                            grip_start_time = current_time
                            place_state = 7
                    elif place_state == 7:  # Wait after closing
                        timeout_reached = (current_time - grip_start_time) > POSITION_TIMEOUT/2
                        if timeout_reached:
                            print("Place sequence complete")
                            place_sequence_active = False
                            place_state = 0

            viewer.sync()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
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