import torch
import os.path
import tempfile
import weakref

from torch.utils.data import Dataset
from torchvision import transforms, utils, io
from typing import Dict, List

import config
import util

class AuxiliaryDataset(Dataset):
    def __init__(self, positive_paths: List[str], negative_paths: List[str],
                 unlabeled_paths: List[str],
                 auxiliary_labels: Dict[str, int], restrict_aux_labels: bool,
                 cache_images_on_disk: bool, data_cache_dir: str,
                 transform=None):
        if restrict_aux_labels:
            unlabeled_paths = []

        self.paths = positive_paths + negative_paths + unlabeled_paths
        self.main_labels = torch.tensor(
            ([1] * len(positive_paths) +
             [0] * len(negative_paths) +
             [-1] * len(unlabeled_paths)),
            dtype=torch.long)
        self.num_aux_classes = 1000
            #len(torch.unique(torch.tensor(list(auxiliary_labels.values()),
            #                              dtype=torch.long)))
        self.aux_labels = torch.tensor(
            [auxiliary_labels.get(os.path.basename(path), -1)
             for path in positive_paths + negative_paths + unlabeled_paths],
            dtype=torch.long)
        self.transform = transform

        # Caching 
        self.cache_images_on_disk: bool = cache_images_on_disk
        self.cache_dir = data_cache_dir
        self.cache_paths = [self._path_to_tmp(path) for path in self.paths]
        self.is_cached = set()

    @property
    def num_auxiliary_classes(self):
        return self.num_aux_classes

    def _path_to_tmp(self, path):
        bpath = os.path.basename(path)
        return os.path.join(self.cache_dir, bpath)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        cache_path = self.cache_paths[index]
        main_label = self.main_labels[index]
        aux_label = self.aux_labels[index]
        if (cache_path in self.is_cached or
            (self.cache_images_on_disk and os.path.isfile(cache_path))):
            # Set path to cache path so we will read from disk
            path = cache_path
            self.is_cached.add(cache_path)

        if path.startswith('http'):
            raw_data = util.download(path)
            # Write to disk if we are caching
            if self.cache_images_on_disk:
                with open(cache_path, 'wb') as f:
                    f.write(raw_data)
                self.is_cached.add(cache_path)

            data = torch.tensor(list(raw_data), dtype=torch.uint8)
            image = io.decode_image(data, mode=io.image.ImageReadMode.RGB)
        else:
            image = io.read_image(path, mode=io.image.ImageReadMode.RGB)
        image = self.transform(image) if self.transform else image
        return image, main_label, aux_label
