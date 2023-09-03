import numpy as np
from PIL import Image
from operator import itemgetter
import albumentations as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

class StreamDataset(Dataset):
    """
    Dataset with sequential sampling
    """
    def __init__(self, data_dict, data_idxs, seq, is_train=True, sample_weights=None):
        super(StreamDataset, self).__init__()
        self.data_dict = data_dict
        self.data_idxs = self._check_idxs(data_idxs)
        self.seq = seq
        self.is_train = is_train
        self.sample_idxs = self._get_sample_idxs()
        self.img_features = self._get_img_files(data_dict)
        self.labels = self._get_img_labels(self.data_idxs)
        self.sample_weights = np.ones(5) if sample_weights is None else sample_weights
        self.sampler = self._get_sampler()

        self.aug = A.Compose([
            A.Resize(300, 300),
            A.Resize(300, 300),
            A.RandomCrop(224, 224),
            # A.RandomBrightness(),
            # A.RandomFog(),
            A.RandomSunFlare(p=0.1),
            A.ColorJitter(),
            A.Flip(),
            A.Normalize(mean=[0.4814745228235294, 0.3937631628627451, 0.32494580560784314],
                        std=[0.244513296627451, 0.2267027840784314, 0.2072741945882353])
        ])

    def __getitem__(self, idx):
        idx = self.sample_idxs[idx]
        # Compose aug input
        img_files = self.img_features[idx-self.seq+1 : idx+1]  # SEQ X DIM

        imgs = []
        for img_file in img_files:
            img = np.array(Image.open(img_file))
            imgs.append(self._aug_imgs(img))
        img = np.stack(imgs, axis=0).transpose((0, 3, 1, 2))

        if self.is_train:
            label = np.array(self.labels[idx - self.seq + 1: idx + 1])[..., np.newaxis]
            return img, label
        else:
            return img

    def __len__(self):

        return len(self.sample_idxs)

    def _check_idxs(self, idx_list):
        if isinstance(idx_list[0], int):
            data_names = list(self.data_dict.keys())
            idx_list = [data_names[idx] for idx in idx_list]
        idx_list = sorted(list(set(idx_list)))

        return idx_list

    def _aug_imgs(self, img):
        img = self.aug(image=img)["image"]

        return img

    def _get_sample_idxs(self):
        sample_idxs = []
        count = 0
        for data_name in self.data_idxs:
            idxs = list(range(count + self.seq, count + len(self.data_dict[data_name]["img"])))
            sample_idxs += idxs
            count += len(self.data_dict[data_name]["img"])

        return sample_idxs

    def _get_img_files(self, data_names):
        img_files = []
        for data_name in data_names:
            img_files += self.data_dict[data_name]["img"]

        return img_files

    def _get_img_labels(self, data_names):
        labels = []
        for data_name in data_names:
            labels += self.data_dict[data_name]["phase"]

        return labels

    def _get_sampler(self):
        if isinstance(self.sample_weights, list):
            self.sample_weights = np.array(self.sample_weights)
        labels = itemgetter(*self.sample_idxs)(self.labels)
        weights = self.sample_weights[list(labels)]
        sampler = WeightedRandomSampler(weights, num_samples=len(self.sample_idxs), replacement=True)

        return sampler