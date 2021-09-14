import io

import numpy as np
from PIL import Image, ImageEnhance
import torch

from typing import Any, Dict, List


torch.set_grad_enabled(False)
torch.set_num_threads(1)


# AUGMENTATIONS


def brightness(image, factor):
    br_enhancer = ImageEnhance.Brightness(image)
    return br_enhancer.enhance(factor)


def contrast(image, factor):
    cn_enhancer = ImageEnhance.Contrast(image)
    return cn_enhancer.enhance(factor)


def flip(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def grayscale(image):
    return image.convert("L")


def resize(image, params):
    return image.resize(((int)(params * image.size[0]), (int)(params * image.size[1])))


def rotate(image, angle):
    return image.rotate(angle)


# INFERENCE


def run(
    image_bytes: bytes,
    image_patch: List[float],
    augmentations: Dict[str, Any],
    input_format: str,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    with io.BytesIO(image_bytes) as image_buffer:
        image = Image.open(image_buffer)

        # Preprocess
        image = image.convert("RGB")

        # Crop
        if image_patch:
            x1f, y1f, x2f, y2f = image_patch
            w, h = image.size
            image = image.crop(
                ((int)(x1f * w), (int)(y1f * h), (int)(x2f * w), (int)(y2f * h))
            )

        # Apply transformations (augmentations is a dict)
        if "flip" in augmentations:
            image = flip(image)
        if "gray" in augmentations:
            image = grayscale(image)
        if "brightness" in augmentations:
            image = brightness(image, augmentations["brightness"])
        if "contrast" in augmentations:
            image = contrast(image, augmentations["contrast"])
        if "resize" in augmentations:
            image = resize(image, augmentations["resize"])
        if "rotate" in augmentations:
            image = rotate(image, augmentations["rotate"])

        image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
        image = image.permute(2, 0, 1)  # HWC -> CHW
        if input_format == "BGR":
            image = torch.flip(image, dims=(0,))  # RGB -> BGR
        image = image.contiguous()
        image = (image - pixel_mean) / pixel_std

    # Input: NCHW
    # Output: {'res4': NCHW, 'res5': NCHW} where N = 1
    output_dict = model(image.unsqueeze(dim=0))
    return {k: v.detach() for k, v in output_dict.items()}
