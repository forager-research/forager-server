import os.path
from dataclasses import dataclass
from pathlib import Path
from urllib.request import Request, urlopen

GIGABYTE = 1024 * 1024 * 1024


def compute_instance_ip() -> str:
    # Get instance IP (https://stackoverflow.com/q/23362887)
    if False:
        ip_request = Request(
            "http://metadata/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip"
        )
        ip_request.add_header("Metadata-Flavor", "Google")
        return urlopen(ip_request).read().decode()
    else:
        return ""


@dataclass
class CloudRunConfig:
    CLOUD_RUN_N_MAPPERS: int = 50
    CLOUD_RUN_N_RETRIES: int = 1


@dataclass
class ClusterConfig:
    TERRAFORM_MODULE_PATH = Path("./terraform").resolve()
    REUSE_EXISTING = True
    MOUNT_DIR = Path("~/.forager/mount").expanduser().resolve()
    CLEANUP_TIME = 20 * 60  # seconds (destroy cluster after idle for this long)


@dataclass
class MapperConfig:
    NUM_RETRIES = 5
    CHUNK_SIZE = lambda nproc: 3
    REQUEST_MULTIPLE = lambda nproc: nproc
    REQUEST_TIMEOUT = (
        3 * 60
    )  # seconds; more than a minute per image is probably too much
    CLOUD_RUN_URL = "https://forager-index-mapper-g6rwrca4fq-uc.a.run.app"


@dataclass
class AdderConfig:
    NUM_RETRIES = 5
    CHUNK_SIZE = lambda nproc: 1
    REQUEST_MULTIPLE = lambda nproc: nproc
    REQUEST_TIMEOUT = 5 * 60  # seconds


@dataclass
class ResizerConfig:
    NUM_RETRIES = 20  # all should succeed
    CHUNK_SIZE = lambda nproc: 20
    REQUEST_MULTIPLE = lambda nproc: nproc
    REQUEST_TIMEOUT = 60  # seconds
    OUTPUT_BUCKET = "foragerml"
    OUTPUT_DIR_TMPL = "thumbnails/{}/"
    MAX_HEIGHT = 200


NUM_IMAGES_TO_MAP_BEFORE_CONFIGURING_INDEX = 100
EMBEDDING_DIM = 2048
INDEX_PCA_DIM = 128
INDEX_SQ_BYTES = 8
INDEX_SUBINDEX_SIZE = 1_000_000

# TODO(mihirg): Toggle when/if FAISS supports merging direct map indexes
BUILD_UNCOMPRESSED_FULL_IMAGE_INDEX = False


@dataclass
class TrainerConfig:
    MAX_RAM = 35 * GIGABYTE
    NER_N_CENTROIDS_MULTIPLE = 39
    ER_EMBEDDING_SAMPLE_RATE = 0.1
    STATUS_ENDPOINT = "/trainer_status"
    STATUS_CALLBACK: str

    def __init__(self, instance_ip):
        self.STATUS_CALLBACK = f"http://{instance_ip}:5000{self.STATUS_ENDPOINT}"


@dataclass
class BGSplitConfig:
    TRAINING_MAX_RAM = 35 * GIGABYTE
    TRAINER_STATUS_ENDPOINT = "/bgsplit_trainer_status"
    TRAINER_STATUS_CALLBACK: str

    MAPPER_NUM_RETRIES = 5
    MAPPER_CHUNK_SIZE = lambda nproc: 32
    MAPPER_REQUEST_MULTIPLE = lambda nproc: nproc
    MAPPER_REQUEST_TIMEOUT = (
        3 * 60
    )  # seconds; more than a minute per image is probably too much
    MAPPER_CLOUD_RUN_URL = "https://forager-bgsplit-mapper-g6rwrca4fq-uc.a.run.app"
    MAPPER_JOB_DIR_TMPL = "shared/dnn_outputs/{}"

    def __init__(self, instance_ip):
        self.TRAINER_STATUS_CALLBACK = (
            f"http://{instance_ip}:5000{self.TRAINER_STATUS_ENDPOINT}"
        )


@dataclass
class AuxConfig:
    PARENT_DIR = Path("~/forager/aux_labels").expanduser().resolve()
    DIR_TMPL = os.path.join(PARENT_DIR, "{}/{}.pickle")
    UPLOAD_GCS_PATH = "gs://foragerml/aux_labels/"  # trailing slash = directory
    GCS_TMPL = os.path.join(UPLOAD_GCS_PATH, "{}/{}.pickle")
    GCS_PUBLIC_TMPL: str

    def __init__(self, gcs_public_root_url):
        self.GCS_PUBLIC_TMPL = os.path.join(
            gcs_public_root_url, "foragerml/aux_labels/{}/{}.pickle"
        )


@dataclass
class ImageListConfig:
    DEFAULT_SPLIT_NAME: str = "train"


@dataclass
class ClipConfig:
    MODEL_NAME: str = "ViT-B/32"


@dataclass
class ResnetConfig:
    pass


@dataclass
class ModelsConfig:
    CLIP: ClipConfig = ClipConfig()
    RESNET: ResnetConfig = ResnetConfig()


@dataclass
class GlobalConfig:
    INSTANCE_IP = compute_instance_ip()
    GCS_PUBLIC_ROOT_URL = "https://storage.googleapis.com/"

    CLOUD_RUN: CloudRunConfig = CloudRunConfig()
    CLUSTER: ClusterConfig = ClusterConfig()
    MAPPER: MapperConfig = MapperConfig()
    ADDER: AdderConfig = AdderConfig()
    RESIZER: ResizerConfig = ResizerConfig()
    TRAINER: TrainerConfig = TrainerConfig(INSTANCE_IP)
    BGSPLIT: BGSplitConfig = BGSplitConfig(INSTANCE_IP)
    AUX: AuxConfig = AuxConfig(GCS_PUBLIC_ROOT_URL)
    IMAGE_LIST: ImageListConfig = ImageListConfig()
    MODELS: ModelsConfig = ModelsConfig()

    SANIC_RESPONSE_TIMEOUT = 10 * 60  # seconds

    LOCAL_INDEX_BUILDING_NUM_THREADS = 10

    MODEL_OUTPUTS_PARENT_DIR = Path("~/forager/model_outputs").expanduser().resolve()
    MODEL_OUTPUTS_UPLOAD_GCS_PATH = (
        "gs://foragerml/models_outputs/"  # trailing slash = directory
    )

    QUERY_PATCHES_PER_IMAGE = 8
    QUERY_NUM_RESULTS_MULTIPLE = 80

    UPLOADED_IMAGE_BUCKET = "foragerml"
    UPLOADED_IMAGE_DIR = "uploads/"

    CLIP_TEXT_INFERENCE_CLOUD_RUN_URL = (
        "https://forager-clip-text-inference-g6rwrca4fq-uc.a.run.app"
    )

    BRUTE_FORCE_QUERY_CHUNK_SIZE = 512
    DEFAULT_QUERY_MODEL = "imagenet"
    EMBEDDING_DIMS_BY_MODEL = {
        "imagenet": 2048,
        "imagenet_early": 1024,
        "imagenet_full": 2048,
        "imagenet_full_early": 1024,
        "clip": 512,
    }
    BGSPLIT_EMBEDDING_DIM = 2048

    EMBEDDING_FILE_NAME = "embeddings.npy"
    MODEL_SCORES_FILE_NAME = "scores.npy"

    DNN_SCORE_CLASSIFICATION_THRESHOLD = 0.5
    ACTIVE_VAL_STARTING_BUDGET = 10
    MIN_TIME_BETWEEN_KEEP_ALIVES = 10  # seconds


CONFIG = GlobalConfig()
