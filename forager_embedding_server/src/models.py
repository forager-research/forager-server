import io
from typing import List, Optional, Tuple

import clip
import numpy as np
import torch
import torchvision.models.resnet
import torchvision.transforms
from PIL import Image

from forager_embedding_server.config import CONFIG
from forager_embedding_server.utils import load_remote_file

torch.set_grad_enabled(False)
torch.set_num_threads(1)


class EmbeddingModel:
    def output_dim(self, *args, **kwargs) -> int:
        raise NotImplementedError("Model must implement output_dim")

    def embed_text(self, text: List[str], *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed text")

    def embed_images(self, images: List[np.ndarray], *args, **kwargs) -> np.ndarray:
        raise NotImplementedError("Model cannot embed images")


class CLIP(EmbeddingModel):
    def __init__(self):
        self.model, self.preprocess = clip.load(
            CONFIG.MODELS.CLIP.MODEL_NAME, device="cpu", jit=True
        )

    def output_dim(self):
        return 512

    def embed_text(self, text, *args, **kwargs):
        text = clip.tokenize(text)
        text_features = self.model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.numpy()

    def embed_images(self, images, *args, **kwargs):
        preprocessed_images = torch.stack(
            [self.preprocess(Image.fromarray(img)) for img in images]
        )
        image_features = self.model.encode_image(preprocessed_images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.numpy()


class TorchvisionResNetFeatureModel(torchvision.models.resnet.ResNet):
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        res4 = self.layer3(x)
        res5 = self.layer4(res4)

        x = self.avgpool(res5)
        x = torch.flatten(x, 1)
        linear = self.fc(x)

        return {"res4": res4, "res5": res5, "linear": linear}


class ResNet(EmbeddingModel):
    def __init__(self):
        # Create model
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 6, 3]
        self.model = TorchvisionResNetFeatureModel(block, layers)

        # Load model weights
        state_dict = torchvision.models.utils.load_state_dict_from_url(
            torchvision.models.resnet.model_urls["resnet50"]
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Setup input pipeline
        self.pixel_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.pixel_std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.input_format = "RGB"

    def output_dim(self, layer: str = "res4"):
        return {"res4": 1024, "res5": 2048, "linear": 1000}[layer]

    def embed_images(
        self,
        images,
        *args,
        image_patches: Optional[List[Tuple[float, float, float, float]]] = None,
        layer: str = "res4",
        input_size: Tuple[int, int] = (256, 256),
        **kwargs,
    ):
        resize_fn = torchvision.transforms.Resize(input_size, antialias=True)
        converted_images = []
        for idx, img in enumerate(images):
            # Preprocess
            image = Image.fromarray(img).convert("RGB")

            # Crop
            if image_patches and len(image_patches) > idx:
                x1f, y1f, x2f, y2f = image_patches[idx]
                w, h = image.size
                image = image.crop(
                    ((int)(x1f * w), (int)(y1f * h), (int)(x2f * w), (int)(y2f * h))
                )

            image = torch.as_tensor(np.asarray(image), dtype=torch.float32)  # -> tensor
            image = image.permute(2, 0, 1)  # HWC -> CHW
            if self.input_format == "BGR":
                image = torch.flip(image, dims=(0,))  # RGB -> BGR
            image = image.contiguous()
            image = (image - self.pixel_mean) / self.pixel_std
            image = resize_fn(image)
            converted_images.append(image)

        # Input: NCHW
        # Output: {'res4': CHW, 'res5': CHW} where N = 1
        inputs = torch.stack(converted_images, dim=0)
        output_dict = self.model(inputs)

        return torch.mean(output_dict[layer], dim=[2, 3]).detach().numpy()
