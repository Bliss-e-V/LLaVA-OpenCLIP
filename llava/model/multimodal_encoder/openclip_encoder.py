from typing import List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import open_clip


class SimpleImageProcessor:
    """
    A simple wrapper for the OpenCLIP preprocessing function,
    providing an interface similar to CLIPImageProcessor.
    """

    def __init__(self, preprocess):

        for tf in preprocess.transforms:
            if isinstance(tf, transforms.Normalize):
                self.image_mean = tf.mean
            elif isinstance(tf, transforms.Resize):
                self.crop_size = {"height": tf.size[0], "width": tf.size[1]}

        self.prepc = preprocess

    def preprocess(self, images, return_tensors="pt", **kwargs):
        if return_tensors != "pt":
            raise NotImplementedError("Only torch.Tensor output is supported.")
        if isinstance(images, list):
            processed = [self.prepc(image).unsqueeze(0) for image in images]
            pixel_values = torch.cat(processed, dim=0)
        else:
            pixel_values = self.prepc(images).unsqueeze(0)
        return {"pixel_values": pixel_values}


class OpenCLIPVisionTower(nn.Module):
    """
    Vision encoder using OpenCLIP.
    Implements a similar API as CLIPVisionTower with support for hidden
    state selection via hooks.
    """

    def __init__(self, vision_tower: str, args, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        # self.pretrained = getattr(args, "vision_tower_pretrained", None)

        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = {}  # Placeholder if you wish to delay loading
            # self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} already loaded, skipping.")
            return

        # Load OpenCLIP model and its preprocessing function.
        self.vision_tower, self.preprocess_fn = open_clip.create_model_from_pretrained(
            self.vision_tower_name
        )
        # Set it to eval mode
        self.vision_tower.eval()  # NOTE: I only added this after both pretraining and finetuning; but Mario tested locally that for hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B it does NOT make a difference :)
        if device_map is not None:
            self.vision_tower.to(device_map)
        # Freeze model parameters.
        for param in self.vision_tower.parameters():
            param.requires_grad = False

        # Wrap preprocess_fn so that it resembles CLIPImageProcessor.
        self.image_processor = SimpleImageProcessor(self.preprocess_fn)
        self.preprocess = self.image_processor.preprocess

        self.is_loaded = True

    def get_hidden_states(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Obtains intermediate hidden states using hooks. Adjust which blocks
        to hook based on your model's architecture.
        """
        hidden_states = []
        hooks = []

        def hook_fn(module, input, output):
            hidden_states.append(output)

        # Register hooks on visual transformer blocks.
        for block in self.vision_tower.visual.transformer.resblocks:
            hook = block.register_forward_hook(hook_fn)
            hooks.append(hook)

        # Run a forward pass to trigger hooks.
        with torch.no_grad():
            _ = self.vision_tower.encode_image(images)

        # Remove hooks.
        for hook in hooks:
            hook.remove()

        return hidden_states

    def feature_select(
        self, hidden_states: List[torch.Tensor], final_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Select features based on configuration. If hidden states are available,
        use the one at select_layer; otherwise, fall back to final output.
        """
        if isinstance(hidden_states, list) and len(hidden_states) > self.select_layer:
            features = hidden_states[self.select_layer]
        else:
            features = final_features

        if self.select_feature == "patch":
            # Assume the first token is CLS and remove it.
            features = features[:, 1:]
        elif self.select_feature == "cls_patch":
            # Return tokens with CLS included.
            pass
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        return features

    @torch.no_grad()
    def forward(self, images: Union[torch.Tensor, list]):
        # if not self.is_loaded:
        #     self.load_model()

        # Use self.preprocess if images are not already tensors.
        if isinstance(images, list) or not torch.is_tensor(images):
            processed = self.preprocess(images)["pixel_values"]
        else:
            processed = images

        images_device = processed.to(self.device, dtype=self.dtype)
        final_features = self.vision_tower.encode_image(images_device)
        final_features = F.normalize(
            final_features, dim=-1
        )  # Not sure if we really need to do this

        # Extract hidden states
        hidden_states = self.get_hidden_states(images_device)
        selected_features = self.feature_select(hidden_states, final_features).to(
            processed.dtype
        )
        return selected_features

    @property
    def dtype(self):
        return (
            next(self.vision_tower.parameters()).dtype
            if self.is_loaded
            else torch.float32
        )

    @property
    def device(self):
        return (
            next(self.vision_tower.parameters()).device
            if self.is_loaded
            else torch.device("cpu")
        )

    # NOTE: I have NOT tested the below getattr functions for OpenCLIP yet. The default
    #  values are the ones for the model I am interested in (ViT-L-14-336)
    @property
    def hidden_size(self):
        return self.vision_tower.visual.ln_post.normalized_shape[
            0
        ]  # 1024 for ViT-L-14

    @property
    def image_size(self):
        # Default image resolution for this model is 336.
        return self.image_processor.crop_size["height"]  # 336 for ViT-L-14

    @property
    def num_patches_per_side(self):
        # Default patch size is 14 so that 336/14 = 24 patches per side.
        patch_size = self.vision_tower.visual.conv1.kernel_size[0]  # 14 for ViT-L-14
        return self.image_size // patch_size

    @property
    def num_patches(self):
        n = self.num_patches_per_side
        return n * n


### TESTS

if __name__ == "__main__":

    import os
    from transformers import AutoConfig, LlamaConfig

    class LlavaConfig(LlamaConfig):
        model_type = "llava_llama"

    AutoConfig.register("llava_llama", LlavaConfig)
    vision_tower_cfg = AutoConfig.from_pretrained(
        os.path.join("llava", "model", "multimodal_encoder"), trust_remote_code=True
    )
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )

    # Use OpenCLIP if the model identifier indicates so:
    vt = OpenCLIPVisionTower(
        vision_tower,
        args=vision_tower_cfg,
    )
    print("debug here")
