#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
from dataclasses import dataclass, field

from sympy import false, true


@dataclass
class MBMConfig:
    """Configuration class for the MBM policy.

    Args:
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
    """
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100
    num_image_tokens: int = 729
    freeze_pretrained: bool = False
    
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.state": [14],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [14],
        }
    )
    
    vision_backbone: str = "siglip"
    pretrained_backbone_weights: str | None = None
    
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"
    
    open_lm_config: dict = field(
        default_factory=lambda: [
            "--model", "79m",
            "--torchcompile",
            "--qk-norm",
        ]
    )
    

    