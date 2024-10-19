from open_lm.model import Transformer, create_params
from open_lm.distributed import world_info_from_env
from open_lm.params import parse_args
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import timm
import torch
import torch.distributed
from transformers import AutoTokenizer
import numpy as np
from huggingface_hub import PyTorchModelHubMixin


from functools import partial
from enum import IntEnum

from torchvision.transforms import (
    Compose,
    Resize,
    InterpolationMode,
    ToTensor,
    Normalize,
)

from lerobot.common.policies.mbm.configuration_mbm import MBMConfig


EMB_IMAGE_SIZE = 384

class OutputType(IntEnum):
    EMBEDDING = 1
    TOKEN = 2


def unpack_tuple(fn):
    def wrapper(*args, **kwargs):
        result = fn(*args, **kwargs)
        return result[0] if (isinstance(result, tuple) or isinstance(result, list))  else result

    return wrapper


class TimmFeaturizer(nn.Module):
    def __init__(
        self, timm_string, img_size=EMB_IMAGE_SIZE, override_forward=True
    ) -> None:
        super().__init__()

        def load_model_helper():
            return timm.create_model(
                timm_string, pretrained=True, num_classes=0, img_size=img_size
            )  # .to(torch.bfloat16)


        if torch.distributed.is_initialized():
            # Need to download on only one rank at a time; start with rank 0, then do the rest.
            local_rank = world_info_from_env()[0]
            if local_rank == 0:
                self.model = load_model_helper()
            torch.distributed.barrier()
            if local_rank != 0:
                self.model = load_model_helper()
        else:
            self.model = load_model_helper()

        self.model.eval()
        self.img_size = img_size
        self.embed_dim = self.model.embed_dim

        model_data_cfg = timm.data.resolve_model_data_config(self.model)
        model_data_cfg["input_size"] = (3, img_size, img_size)

        if override_forward:
            self.model.forward = unpack_tuple(
                partial(
                    self.model.get_intermediate_layers,
                    n={len(self.model.blocks) - 2},
                )
            )

    @torch.inference_mode
    def forward(self, x):
        # [batch, num_patches, feature_dim]
        return self.model(x)


class SiglipFeaturizer(TimmFeaturizer):
    def __init__(self, img_size=EMB_IMAGE_SIZE, override_forward=True) -> None:
        super().__init__("vit_so400m_patch14_siglip_384", img_size, override_forward)


SIGLIP_TRANFORM = Compose(
    [
        Resize(
            size=(EMB_IMAGE_SIZE, EMB_IMAGE_SIZE),
            interpolation=InterpolationMode.BICUBIC,
            max_size=None,
            antialias=True,
        ),
        Normalize(mean=(0.5000, 0.5000, 0.5000), std=(0.5000, 0.5000, 0.5000)),
    ]
)


EMBEDDING_REGISTRY = {
    "siglip": {
        "cls": SiglipFeaturizer,
        "transform": SIGLIP_TRANFORM,
        "output_type": OutputType.EMBEDDING,
    }
}

class StateTokenizer():
    def __init__(self, tokenizer_name: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, state: torch.Tensor):
        shape = state.shape
        if state.ndim > 2:
            state = state.view(-1, shape[-1])
        batch_size = state.shape[0]
        state_dim = state.shape[-1]
            
        assert state.dtype == torch.uint8, "State tensor must be uint8"
        states_text = []
        for i in range(batch_size):
            state_text = " ".join([f"{state[i, j].item()}" for j in range(state_dim)])
            states_text.append(state_text)
        
        tokens = self.tokenizer(states_text, return_tensors="pt")
        return tokens.input_ids.view(shape)
    
    @staticmethod
    def _convert_to_int(unique_action_str: str):
        try:
            return int(unique_action_str)
        except ValueError:
            return 0
    
    def decode(self, tokens: torch.Tensor):
        shape = tokens.shape
        text_out = self.tokenizer.batch_decode(tokens.flatten())
        actions_str = [action.split(" ") for action in text_out]
        actions_list = torch.tensor([self._convert_to_int(a) for action in actions_str for a in action])
        if len(actions_list) > np.prod(shape):
            actions_list = actions_list[:np.prod(shape)]
        elif len(actions_list) < np.prod(shape):
            actions_list = torch.cat([actions_list, torch.zeros(mul(shape) - len(actions_list), dtype=actions_list.dtype)], dim=-1)
        return actions_list.view(shape)

class MbmProjector(nn.Module):
    def __init__(self, input_vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.hidden_dim = input_vision_dim * 4
        self.projector = nn.Sequential(
            nn.Linear(input_vision_dim, self.hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )

    def forward(self, image_featues):
        # assuming no interleaving for now so [b, patch, dim]
        return self.projector(image_featues)


class MbmTransformer(Transformer):
    def __init__(self, config: MBMConfig):
        open_lm_config = parse_args(config.open_lm_config)
        params = create_params(open_lm_config)
        super().__init__(params)

        self.freeze_pretrained = config.freeze_pretrained
        self.state_dim = config.input_shapes["observation.state"]
        self.num_image_tokens = config.num_image_tokens
        self.image_extractors = torch.nn.ModuleList([
            EMBEDDING_REGISTRY["siglip"]["cls"]()
        ])
        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        self.image_dim = sum([p.embed_dim for p in self.image_extractors])*len(self.expected_image_keys)
        self.image_projector = MbmProjector(self.image_dim, self.dim)
        # self.start_token_id = self.tokenizer.bos_token_id
        self.sep_token_id = 50277
        # self.pad_token_id = self.tokenizer.pad_token_id
        self.image_dim = sum([p.embed_dim for p in self.image_extractors])


        if self.freeze_pretrained:
            print("freezing pre-trained models")
            for m in self.image_extractors:
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]):
        """
        Args:
            batch: A dictionary containing the input tensors.
        """
        # Extract necessary inputs from the batch
        action_ids = batch.get("action_ids")
        past_key_values = batch.get("past_key_values", None)
        use_cache = batch.get("use_cache", False)
        attention_mask = batch.get("attention_mask", None)
        
        # Run transforms on the images
        images = batch.get("observation.images")
        
        image_embeddings = torch.cat([self.image_extractors[i](image) for i in range(len(self.image_extractors)) for image in images], dim=-1)
        image_embeddings = self.image_projector(image_embeddings)
        
        # Tokenize the task description and the state
        # task_description = batch.get("observation.task_description", "")
        # task_description_tokens = self.tokenizer(task_description, return_tensors="pt")
        
        # breakpoint() # TODO: pad the length of the task description and state to be the same
        # tokens = torch.cat([self.sep_token_id, task_description_tokens.input_ids, self.sep_token_id, state_tokens.input_ids], dim=1)
        action_shape = action_ids.shape
        action_ids = action_ids.reshape(action_shape[0], -1)
        separator_tokens = torch.full_like(action_ids[:, 0:1], self.sep_token_id)
        tokens = torch.cat([separator_tokens, action_ids], dim=1).to(image_embeddings.device)
                                                                    
        embeddings = self.tok_embeddings(tokens)

        x = torch.cat([image_embeddings, embeddings], dim=1)

        x = self.post_embed_norm(x)

        if past_key_values is None:
            past_key_values = [None] * self.n_layers
        elif isinstance(past_key_values, tuple):
            past_key_values = list(past_key_values)

        for i, layer in enumerate(self.layers):
            if self.grad_checkpointing:
                x, past_key_values[i] = checkpoint(
                    layer, x, past_key_values[i], use_cache, attention_mask
                )
            else:
                x, past_key_values[i] = layer(
                    x,
                    past_key_values[i],
                    use_cache=use_cache,
                    attention_mask=attention_mask,
                )

        if past_key_values[0] is None:
            past_key_values = None

        x = self.norm(x)
        output = self.output(x)
        
        # extract the state tokens
        # TODO: when adding task descriptions, fix to work with variable length task descriptions
        action_tokens = output[:, self.num_image_tokens + 1:]
        action_tokens = action_tokens.view(*action_shape, -1)
        x = x[:, self.num_image_tokens + 1:].view(*action_shape, -1)
        
        # Follow llama in casting this to float
        return action_tokens.float(), x, past_key_values


def create_model(args, directives=["siglip", "dino"]):
    model = MbmTransformer(create_params(args), directives=directives)
    return model


class MBMPolicy( 
    nn.Module,
    PyTorchModelHubMixin,
    library_name="lerobot",
    repo_url="https://github.com/huggingface/lerobot",
    tags=["robotics", "mbm"],
    ):
    name = "mbm"

    def __init__(
        self,
        config: MBMConfig | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = MBMConfig()
        self.config: MBMConfig = config

        self.model = MbmTransformer(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        
        # Initialize the image transform
        self.image_transform = SIGLIP_TRANFORM
        self.state_tokenizer = StateTokenizer(config.tokenizer_name)
        
        # Cross entropy loss with logits
        self.loss_fn = nn.CrossEntropyLoss()

        self.reset()
        
    def process_batch(self, batch: dict[str, torch.Tensor]):
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy
            batch["observation.images"] = [self.image_transform(batch[k]) for k in self.expected_image_keys]
            
        state_int8 = (batch["observation.state"] / 360 * 255 + 128).to(torch.uint8)
        batch["observation.state_ids"] = self.state_tokenizer(state_int8)
        action_int8 = (batch["action"] / 360 * 255 + 128).to(torch.uint8)
        batch["action_ids"] = self.state_tokenizer(action_int8)
        return batch
        
    def process_action(self, action_logits: torch.Tensor):
        action_ids = torch.argmax(action_logits, dim=-1)
        actions_int8 = self.state_tokenizer.decode(action_ids)
        actions = (actions_int8.to(torch.float) - 128) / 256 * 360 
        return actions
    
    def reset(self):
        # Implement reset logic if needed
        pass

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()

        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # shallow copy
            transformed_images = [self.image_transform(batch[k]) for k in self.expected_image_keys]
            batch["observation.images"] = torch.stack(transformed_images, dim=-4)

        actions, _, _ = self.model(batch)

        return actions[:, 0]  # Return the first action in the sequence

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                 
        batch = self.process_batch(batch)
        actions_hat_ids, _, _ = self.model(batch)
        
        loss  = self.loss_fn(actions_hat_ids.reshape(-1, actions_hat_ids.shape[-1]), batch["action_ids"].reshape(-1).to(actions_hat_ids.device))
        
        actions_hat = self.process_action(actions_hat_ids)[:, :batch["action_ids"].shape[1]]
        

        l1_loss = nn.functional.l1_loss(batch["action"].to(actions_hat.device), actions_hat, reduction="none")
        l1_loss = (l1_loss * ~batch["action_is_pad"].unsqueeze(-1).to(actions_hat.device)).mean()

        loss_dict = {
            "l1_loss": l1_loss.item(),
            "loss": loss,
        }

        return loss_dict


if __name__ == "__main__":
    import torch
    from PIL import Image
    import numpy as np
    
    # Test StateTokenizer
    print("Testing StateTokenizer...")
    state_tokenizer = StateTokenizer("gpt2")
    sample_state = torch.randint(0, 256, (2, 10), dtype=torch.uint8)
    state_tokens = state_tokenizer(sample_state)
    print(f"State tokens shape: {state_tokens.shape}")

    # Test SiglipFeaturizer
    print("\nTesting SiglipFeaturizer...")
    siglip = SiglipFeaturizer()
    dummy_image = torch.from_numpy(np.random.rand(3, EMB_IMAGE_SIZE, EMB_IMAGE_SIZE)).float()
    transformed_image = SIGLIP_TRANFORM(dummy_image).unsqueeze(0)
    image_features = siglip(transformed_image)
    print(f"Image features shape: {image_features.shape}")

    # Test MBMPolicy
    print("\nTesting MBMPolicy...")
    config = MBMConfig(
        input_shapes={"observation.state": 10, "observation.images": (3, 480, 640)},
        tokenizer_name="gpt2"
    )
    policy = MBMPolicy(config)

    # Create a dummy batch
    batch = {
        "observation.state": torch.randn(2, 10),
        "observation.images": torch.from_numpy(np.random.rand(2,3, 480, 640)).float(),
        "action": torch.randn(2, 10),
        "action_is_pad": torch.zeros(2, dtype=torch.bool)
    }

    # Process the batch and run forward pass
    loss_dict = policy(batch)

    print(f"Loss: {loss_dict['loss'].item()}")
    print(f"L1 Loss: {loss_dict['l1_loss']}")

    print("\nTests completed.")
