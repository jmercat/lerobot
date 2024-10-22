from dataclasses import dataclass, field

@dataclass
class GeminiConfig:
    """Configuration class for the Gemini-based policy."""

    # Input / output structure
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

    # Remove input normalization modes
    # Keep output normalization modes for action scaling
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action": "mean_std",
        }
    )

    # Gemini API configuration
    gemini_api_key: str = "YOUR_GEMINI_API_KEY_HERE"

    # Task description
    task_description: str = "perform the next step in the current task"
