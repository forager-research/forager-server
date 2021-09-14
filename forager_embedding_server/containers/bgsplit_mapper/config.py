import functools
import numpy as np


GCS_URL_PREFIX = "https://storage.googleapis.com"
DOWNLOAD_NUM_RETRIES = 3

BATCH_SIZE = 128
DATA_FILE_TMPL = "/shared/dnn_outputs/{}/{}/{}-{{}}.npy"
