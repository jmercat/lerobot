# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache

import cv2
import torch
from deepdiff import DeepDiff
from termcolor import colored
import numpy as np

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method


def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True


def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def display_status_overlay(image, text, duration=1.0):
    """
    Add a text overlay to an image.
    
    Args:
        image: The camera image to overlay text on (numpy array)
        text: Text to display
        duration: How long the text should be visible in seconds
    
    Returns:
        The image with text overlay
    """
    # Don't modify the original image
    overlay_img = image.copy()
    
    # Calculate text dimensions for proper positioning
    h, w = overlay_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    # Position text in the top center
    x = (w - text_size[0]) // 2
    y = text_size[1] + 20  # Margin from top
    
    # Draw semi-transparent background for text
    sub_img = overlay_img[y-text_size[1]-10:y+10, x-10:x+text_size[0]+10]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    overlay_img[y-text_size[1]-10:y+10, x-10:x+text_size[0]+10] = cv2.addWeighted(
        sub_img, 0.5, white_rect, 0.5, 0
    )
    
    # Draw text
    cv2.putText(
        overlay_img, 
        text, 
        (x, y), 
        font, 
        font_scale, 
        (0, 255, 0),  # Green text
        thickness
    )
    
    return overlay_img


@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
    status_message: str = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    # Variables to track status display
    status_start_time = None
    status_duration = 1.0  # How long to display status messages, in seconds
    
    # If a status message is provided, set its start time
    if status_message:
        status_start_time = time.perf_counter()
        logging.info(f"Status: {status_message}")

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

            if policy is not None:
                pred_action = predict_action(
                    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
                )
                # Action can eventually be clipped using `max_relative_target`,
                # so action actually sent is saved in the dataset.
                action = robot.send_action(pred_action)
                action = {"action": action}

        if dataset is not None:
            frame = {**observation, **action, "task": single_task}
            dataset.add_frame(frame)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                img = observation[key].numpy()
                
                # Add overlay text if there's an active status message
                if status_message and status_start_time is not None:
                    current_time = time.perf_counter()
                    # Check if the status message should still be displayed
                    if current_time - status_start_time < status_duration:
                        img = display_status_overlay(img, status_message, status_duration)
                    else:
                        # Clear status after duration expires
                        status_message = None
                        status_start_time = None
                        
                cv2.imshow(key, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps, display_cameras=False):
    # Ensure the robot stops following the leader arm
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()
    
    # For SO100 robots, explicitly stop movement
    if robot.robot_type.lower() == 'so100' and hasattr(robot, 'stop_following_leader'):
        robot.stop_following_leader()
    
    # Force mode to stop if possible
    if hasattr(robot, 'set_control_mode'):
        robot.set_control_mode('position')  # Switch to position mode to stop
    
    # Send zero velocities if possible
    if hasattr(robot, 'follower_arms') and robot.follower_arms:
        for arm_name, arm in robot.follower_arms.items():
            if hasattr(arm, 'write') and 'Goal_Velocity' in arm.motors:
                try:
                    # Try to set velocities to zero
                    zeros = [0.0] * len(arm.motors['Goal_Velocity'])
                    arm.write('Goal_Velocity', zeros)
                except Exception as e:
                    logging.warning(f"Failed to set zero velocities: {e}")
    
    # Let the robot settle for a moment
    time.sleep(0.1)
    
    # Now allow manual teleoperation during reset
    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
        display_cameras=display_cameras,
        status_message="Reset Time - Position Robot",
    )


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    fps,
    single_task,
    episode_number=None,
):
    status_message = None
    if episode_number is not None:
        status_message = f"Recording Episode {episode_number}"
    
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
        status_message=status_message,
    )
