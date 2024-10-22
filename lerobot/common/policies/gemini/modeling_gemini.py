import torch
from torch import nn, Tensor
from typing import Dict, List
import google.generativeai as genai
from PIL import Image
import io
import random
import re
from lerobot.common.policies.gemini.configuration_gemini import GeminiConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize

import numpy as np
from scipy import interpolate

class GeminiPolicy(nn.Module):
    """
    Gemini-based Policy that uses the Gemini chat API for action generation.
    """

    name = "gemini"

    def __init__(
        self,
        config: GeminiConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = GeminiConfig()
        self.config: GeminiConfig = config

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        # Initialize Gemini
        genai.configure(api_key=config.gemini_api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        # self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.history_length = 6000
        self.action_chunks = 30
        self.chunck_history_length = self.history_length // self.action_chunks
        self.action_downscale = 5
        self.image_history = []
        self.reset()

    def reset(self):
        """Reset the policy state."""
        self.generated_actions = []
        full_action_history = [
            [-35,  81, 111,  25,  17,  -3],
            [-36,  78, 110,  26,  16,   6],
            [-35,  77, 110,  27,  14,  17],
            [-35,  76, 110,  27,  13,  25],
            [-33,  76, 111,  25,  11,  38],
            [-32,  77, 110,  25,  11,  38],
            [-32,  77, 110,  26,  10,  38],
            [-32,  80, 111,  25,  10,  37],
            [-32,  82, 111,  26,   9,  36],
            [-33,  85, 112,  27,   9,  36],
            [-33,  88, 112,  27,  10,  35],
            [-33,  93, 114,  29,  10,  35],
            [-29, 115, 121,  37,   9,  34],
            [-27, 117, 122,  37,   8,  34],
            [-25, 119, 123,  38,   8,  34],
            [-21, 123, 125,  38,   6,  34],
            [-16, 124, 127,  37,   3,  34],
            [-12, 126, 129,  37,   2,  34],
            [ -8, 127, 132,  35,   0,  34],
            [ -3, 128, 135,  33,  -1,  34],
            [ -2, 127, 128,  52,   1,  33],
            [ -1, 126, 129,  50,   0,  34],
            [  0, 123, 124,  54,  -0,  34],
            [  2, 120, 122,  56,  -0,  33],
            [  4, 118, 121,  56,  -2,  33],
            [  6, 114, 118,  57,  -3,  33],
            [  8, 110, 115,  58,  -4,  33],
            [ 12, 107, 115,  55,  -6,  33],
            [ 13, 104, 113,  56,  -7,  33],
            [ 14, 102, 112,  56,  -8,  33],
            [ 17,  99, 110,  55, -11,  32],
            [ 19,  97, 109,  54, -11,  32],
            [ 19,  96, 108,  54, -12,  31],
            [ 21,  93, 105,  54, -12,  32],
            [ 22,  91, 104,  53, -13,  30],
            [ 22,  90, 104,  54, -13,  31],
            [ 24,  87, 102,  53, -14,  30],
            [ 25,  87, 101,  52, -14,  31],
            [ 25,  85, 101,  52, -14,  29],
            [ 26,  84, 100,  52, -15,  30],
            [ 26,  84, 100,  51, -15,  31],
            [ 26,  82,  99,  51, -15,  31],
            [ 27,  81,  98,  51, -15,  31],
            [ 28,  80,  97,  51, -15,  31],
            [ 28,  79,  97,  50, -15,  32],
            [ 28,  78,  96,  50, -15,  33],
            [ 29,  76,  95,  49, -15,  33],
            [ 29,  75,  95,  49, -15,  34],
            [ 29,  74,  95,  48, -16,  34],
            [ 29,  73,  94,  48, -16,  34],
            [ 29,  73,  94,  48, -16,  34],
            [ 29,  71,  93,  47, -16,  34],
            [ 30,  69,  93,  46, -16,  34],
            [ 30,  69,  92,  46, -16,  34],
            [ 29,  68,  92,  46, -16,  34],
            [ 29,  68,  93,  45, -16,  34],
            [ 30,  66,  92,  44, -17,  34],
            [ 29,  66,  92,  44, -16,  34],
            [ 29,  64,  92,  44, -17,  34],
            [ 29,  64,  92,  43, -17,  34],
            [ 29,  63,  93,  43, -17,  34],
            [ 29,  62,  92,  43, -17,  33],
            [ 28,  62,  93,  43, -17,  33],
            [ 28,  62,  93,  43, -17,  33],
            [ 28,  61,  93,  43, -17,  32],
            [ 27,  61,  94,  43, -17,  31],
            [ 27,  61,  94,  43, -17,  30],
            [ 27,  61,  95,  42, -17,  27],
            [ 27,  61,  94,  43, -16,  23],
            [ 27,  61,  94,  44, -16,  14],
            [ 27,  62,  94,  44, -15,   7],
            [ 27,  63,  94,  44, -14,   0],
            [ 27,  64,  95,  43, -13,  -9],
            [ 26,  66,  96,  44, -13, -13],
            [ 26,  67,  95,  45, -13, -15],
            [ 26,  69,  95,  46, -12, -16],
            [ 25,  71,  96,  48, -11, -16],
            [ 24,  73,  95,  51, -10, -17],
            [ 23,  77,  94,  54, -10, -16],
            [ 21,  81,  95,  55,  -8, -16],
            [ 16,  89,  95,  59,  -5, -16],
            [ 13,  93,  94,  62,  -2, -16],
            [  8,  98,  95,  62,   2, -15],
            [  2, 102,  94,  64,   6, -15],
            [ -2, 103,  94,  62,  10, -15],
            [ -7, 104,  94,  60,  14, -15],
            [-11, 104,  94,  58,  17, -15],
            [-14, 104,  94,  56,  18, -14],
            [-20, 101,  95,  49,  21, -14],
            [-20, 102,  95,  49,  21, -14],
            [-23, 100,  97,  45,  22, -14],
            [-28,  95,  99,  39,  23, -14],
            [-31,  92, 100,  35,  23, -14],
            [-32,  91, 101,  34,  23, -14],
            [-34,  88, 102,  32,  22, -14],
            [-35,  86, 102,  30,  22, -14],
            [-35,  84, 103,  29,  22, -14],
            [-36,  83, 104,  28,  21, -14],
            [-36,  82, 105,  27,  21, -13],
            [-36,  81, 106,  27,  20, -12],
            [-35,  81, 107,  25,  20,  -8],
            [-35,  81, 109,  24,  18,  -1],
            [-35,  79, 109,  24,  18,   5],
            [-32,  82, 115,  21,  13,  20],
            [-32,  82, 115,  21,  12,  25],
            [-30,  85, 117,  21,  11,  28],
            [-29,  86, 117,  22,  10,  31],
            [-27,  89, 119,  22,   9,  32],
            [-25,  93, 122,  22,   8,  33],
            [-25,  95, 122,  23,   8,  33],
            [-23,  97, 124,  23,   7,  34],
            [-22, 100, 126,  23,   6,  34],
            [-19, 103, 130,  21,   5,  34],
            [-18, 105, 132,  20,   3,  34],
            [-16, 108, 134,  20,   2,  34],
            [-13, 110, 137,  18,   1,  34],
            [-11, 112, 139,  17,  -0,  34],
            [-10, 113, 140,  17,  -0,  34],
            [ -7, 115, 144,  14,  -2,  34],
            [ -4, 116, 147,  12,  -4,  34],
            [-11, 130, 139,  29,   2,  34],
            [ -7, 130, 141,  28,   1,  34],
            [ -3, 130, 142,  27,  -1,  34],
            [ -0, 129, 144,  27,  -2,  34],
            [  2, 129, 147,  23,  -4,  34],
            [  4, 129, 149,  19,  -6,  34],
            [  6, 129, 153,  14,  -8,  34],
            [  7, 129, 158,   4, -10,  34],
            [  8, 128, 162,  -2, -11,  34],
            [  8, 127, 167, -11, -13,  34],
            [  9, 126, 169, -15, -13,  34],
            [  9, 126, 173, -20, -14,  34],
            [  9, 125, 175, -21, -14,  34],
            [  9, 126, 176, -20, -14,  34],
            [  9, 125, 178, -19, -14,  34],
            [  9, 126, 178, -18, -14,  34],
            [ 10, 127, 179, -14, -14,  34],
            [ 10, 127, 179, -12, -13,  34],
            [ 10, 128, 179, -11, -13,  33],
        ]

        # Downsample the action history
        self.initial_action_history = full_action_history[::self.action_downscale]
        # Trim to the last downsampled actions if necessary
        self.action_history = [[int(x) for x in action] for action in self.initial_action_history[-self.history_length // self.action_downscale:]]
        self.generated_actions = []


    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        if not self.generated_actions:
            self.generate_actions(batch)

        action = self.generated_actions.pop(0)
        action = torch.tensor(action)

        # Update action history
        self.action_history.append([int(x) for x in action.tolist()])
        if len(self.action_history) > self.history_length // self.action_downscale:
            self.action_history.pop(0)

        return action

    def generate_actions(self, batch: dict[str, Tensor]):
        """Generate a new set of actions using Gemini."""
        # Prepare the image history
        current_image = batch.get("observation.images.phone", [])

        self.image_history.append(Image.fromarray((current_image.squeeze(0) * 256).permute(1, 2, 0).byte().cpu().numpy()))
        if len(self.image_history) > self.chunck_history_length:
            self.image_history.pop(0)

        # Prepare the initial prompt  ,
        content_list = [self.image_history[0]]
        initial_prompt = f"You are given the control of a robot. It is controlled with numbers. Each value is an absolute joint angle in degrees of the robot from the base to the end effector" \
             " [base (left 90 right -90), shoulder (up 90 down 0), elbow (extended 0 retract 180), wrist (up -90 down 90), gripper (rotated left-90 to the right 90), gripper (open 53 close 0)],"\
             " the position [ 0, 120, 180, 0, 0,  33] is the initial position of the robot, retracted at the base and the gripper semi-open.\n" \
             "Here is an example of a sequence of actions where the robot is \n" \
             "Picking up an object from one location (on the right side).\n" \
             "Moving it to another location (toward the left side).\n" \
             "Placing the object down at the new location.\n" \
             "Returning to its initial position to possibly repeat the task.\n" \
             "The action sequence is as follows:\n"
        
        for action in self.initial_action_history:
            initial_prompt += f"{action}\n"
        
        initial_prompt += f"\nHistory of images and actions you took:\n"

        # Create a list to hold all content for the generate_content call
        content_list += [initial_prompt]

        # Add image-action pairs to the content list
        for i in range(min(len(self.image_history) - 1, len(self.action_history) // self.action_chunks)):
            content_list.append(self.image_history[i])
            start_idx = i * self.action_chunks
            end_idx = start_idx + self.action_chunks
            action_prompt = ""
            for action in self.action_history[start_idx:end_idx]:
                action_prompt += f"{action}\n"
            action_prompt += "\n"
            content_list.append(action_prompt)

        # Add the current image
        content_list.append(self.image_history[-1])

        # Add the final prompt
        final_prompt = f"Generate the next {self.action_chunks // self.action_downscale} actions for the robot to {self.config.task_description}. " \
                       f"Respond with exactly {self.action_chunks // self.action_downscale} lines, each containing {self.config.output_shapes['action'][0]} int values separated by commas. " \
                       f"Each line should represent an action that significantly moves the robot to a new position. Lines should be different from each other." \
                       f"Make the chunk of actions solve a specific subgoal such that at the end of it the image that you see is the goal image of the subgoal." \
                       f"You can then decide to retry that subgoal or move towards the next subgoal. First describe the subgoal, then the actions to achieve it."
        content_list.append(final_prompt)

        print(f"Length of content list: {len(content_list)}")
        # Generate actions using Gemini
        max_attempts = 3
        for attempt in range(max_attempts):
            response = self.model.generate_content(content_list)
            print(response.text)
            action_str = response.text
            # replace all characters that are not a digit, sign or comma or end of line with an empty string
            action_str = re.sub(r'[^0-9+\-,\n]', '', action_str)
            actions_str = action_str.strip().split('\n')
            print(f"Actions string: {actions_str}")
            # Parse the action strings to a list of lists
            try:
                actions = [[int(x) for x in action_str.split(',')]for action_str in actions_str ]

                if len(actions) > self.action_chunks // self.action_downscale:
                    actions = actions[:self.action_chunks // self.action_downscale]
                elif len(actions) < self.action_chunks // self.action_downscale:
                    actions = actions + actions[:self.action_chunks // self.action_downscale-len(actions)]
                # Validate the generated actions
                if len(actions) == self.action_chunks // self.action_downscale and all(len(action) == self.config.output_shapes['action'][0] for action in actions):
                    downsampled_actions = actions
                    print(f"Valid actions: {downsampled_actions}")
                    break
                else:
                    print(f"Invalid number of actions or action dimensions. Attempt {attempt + 1}/{max_attempts}")
            except ValueError:
                print(f"Error parsing Gemini response. Attempt {attempt + 1}/{max_attempts}")

                if attempt == max_attempts - 1:
                    # If all attempts fail, generate random actions
                    print("Failed to generate valid actions. Using random actions instead.")
                    downsampled_actions = [
                        [9, 131, 179, -8, -11, 34]
                        for _ in range(self.action_chunks // self.action_downscale)
                    ]

        # Interpolate actions
        x = np.arange(0, len(downsampled_actions))
        x_new = np.linspace(0, len(downsampled_actions) - 1, len(downsampled_actions) * self.action_downscale)
        interpolated_actions = []

        for i in range(len(downsampled_actions[0])):
            y = [action[i] for action in downsampled_actions]
            f = interpolate.interp1d(x, y, kind='linear')
            interpolated_actions.append(f(x_new))

        # Transpose and round the interpolated actions
        self.generated_actions = np.round(np.array(interpolated_actions).T).tolist()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        This method is not used for inference in this policy, but is kept for compatibility.
        """
        raise NotImplementedError("Forward pass is not implemented for GeminiPolicy")



