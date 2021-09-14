import torch
import clip

from typing import List

from knn import utils
from knn.mappers import Mapper

import config


torch.set_grad_enabled(False)
torch.set_num_threads(1)


class TextEmbeddingMapper(Mapper):
    def initialize_container(self):
        self.model, _ = clip.load(config.CLIP_MODEL, device="cpu")

    @utils.log_exception_from_coro_but_return_none
    async def process_chunk(
        self, chunk: List[str], job_id, job_args, request_id
    ) -> List[str]:
        with torch.no_grad():
            text = clip.tokenize(chunk)
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return list(map(utils.numpy_to_base64, text_features.numpy()))

    async def process_element(self, *args, **kwargs):
        raise NotImplementedError()


app = TextEmbeddingMapper().server
