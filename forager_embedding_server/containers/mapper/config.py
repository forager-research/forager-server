from detectron2.config.config import get_cfg as get_default_detectron_config
import functools
import numpy as np

WEIGHTS_PATH = "R-50.pkl"  # model will be downloaded here during container build
RESNET_CONFIG = get_default_detectron_config()
RESNET_CONFIG.MODEL.RESNETS.OUT_FEATURES = ["res5"]

GCS_URL_PREFIX = "https://storage.googleapis.com"
DOWNLOAD_NUM_RETRIES = 3

EMBEDDINGS_FILE_TMPL = "/shared/embeddings/{}/{}/{}-{{}}.npy"
REDUCTIONS = {
    None: lambda x: x,
    "average": functools.partial(np.mean, axis=0, keepdims=True),
}
