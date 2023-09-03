import logging
import os
import random

import numpy as np
from operator import itemgetter
from glob import glob
import albumentations as A
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

from utils.WBEmulator import WBEmulator

class ESDDataset(Dataset):
    """
    Frame-wise random sampling
    """  # ESDDataset(data_dict=data_dict, data_idxs=args.train_names + args.val_names + args.test_names, is_train=False, get_name=True)
    def __init__(self, data_dict, data_idxs, is_train=True, get_name=False, class_weights=None, has_label=True):
        self.data_dict = data_dict
        self.data_idxs = self._check_idxs(data_idxs)
        self.is_train = is_train
        self.get_name = get_name
        self.has_label = has_label
        self.img_files = self.get_data("img")
        self.phase_labels = self.get_data("phase") if has_label else None
        self.class_weights = np.ones(5) if class_weights is None else class_weights
        self.sampler = self._get_weighted_sampler()
        self.wber = WBEmulator()
        if is_train:
            self.augs = T.Compose([
                # T.RandAugment(num_ops=5),
                T.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.Resize([250, 250]),
                T.RandomCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.4814745228235294, 0.3937631628627451, 0.32494580560784314],
                            std=[0.244513296627451, 0.2267027840784314, 0.2072741945882353])
            ])
            # self.augs = A.Compose([
            #     A.Resize(250, 250),
            #     A.RandomCrop(224, 224),
            #     # A.RandomBrightness(),
            #     # A.RandomFog(),
            #     # A.RandomSunFlare(p=0.1),
            #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            #     A.Flip(),
            #     A.Normalize()
            # ])
        else:
            self.augs = T.Compose([
                T.Resize([224, 224]),
                T.ToTensor(),
                T.Normalize(mean=[0.4814745228235294, 0.3937631628627451, 0.32494580560784314],
                            std=[0.244513296627451, 0.2267027840784314, 0.2072741945882353])
            ])

    # mean = [0.4814745228235294, 0.3937631628627451, 0.32494580560784314],
    # std = [0.244513296627451, 0.2267027840784314, 0.2072741945882353]

    def __getitem__(self, idx):
        # logging.info(self.img_files[idx])
        img = Image.open(self.img_files[idx])

        if self.is_train:
            if random.random() > 0.8:
                img = self.wber.single_image_processing(img)  # Data augmentation for histogram

        img = self.augs(img)
        # logging.info(img.shape)
        if self.has_label:
            label = np.array([int(self.phase_labels[idx])])
            if self.get_name:
                return img, label, os.path.basename(self.img_files[idx]).split(".")[0]
            else:
                return img, label
        else:
            return img, 0, os.path.basename(self.img_files[idx]).split(".")[0]

    def __len__(self):

        return len(self.img_files)

    def _check_idxs(self, idx_list):
        if isinstance(idx_list[0], int):
            data_names = list(self.data_dict.keys())
            idx_list = [data_names[idx] for idx in idx_list]

        idx_list = sorted(list(set(idx_list)))

        return idx_list

    def get_data(self, header):
        data = []
        for data_idx in self.data_idxs:
            # print("----" * 10, data_idx, header)
            data += self.data_dict[data_idx][header]

        return data

    def read_img(self, idx):
        img_file = self.img_files[idx]
        img = np.array(Image.open(img_file))
        # if img.max() > 1:
        #     img = (img / 255.0).astype(np.float32)

        return img

    def _get_weighted_sampler(self):
        if isinstance(self.class_weights, list):
            self.class_weights = np.array(self.class_weights)
        weights = self.class_weights[self.phase_labels]
        sampler = WeightedRandomSampler(weights, num_samples=weights.size, replacement=True)

        return sampler


class FeatureDataset(Dataset):
    """
    Dataset with sequential sampling
    """
    def __init__(self, data_dict, data_idxs, data_features, seq, is_train=True, sample_weights=None):
        super(FeatureDataset, self).__init__()
        self.data_dict = data_dict
        self.data_idxs = self._check_idxs(data_idxs)
        self.seq = seq
        self.is_train = is_train
        self.sample_idxs = self._get_sample_idxs()
        self.labels = self._get_img_labels(self.data_idxs)
        self.sample_weights = np.ones(5) if sample_weights is None else sample_weights
        self.sampler = self._get_sampler()
        self.img_features = self._get_img_features(data_features)

    def __getitem__(self, idx):
        idx = self.sample_idxs[idx]
        # Compose aug input
        img_features = self.img_features[idx-self.seq+1 : idx+1]  # SEQ X DIM
        label = np.array(self.labels[idx-self.seq+1 : idx+1])
        if self.is_train:
            return img_features, label
        else:
            return img_features

    def __len__(self):

        return len(self.sample_idxs)

    def _get_sample_idxs(self):
        sample_idxs = []
        count = 0
        for data_name in self.data_idxs:
            idxs = list(range(count + self.seq, count + len(self.data_dict[data_name]["img"])))
            sample_idxs += idxs
            count += len(self.data_dict[data_name]["img"])

        return sample_idxs

    def _get_img_features(self, data_embs):
        emb_all = []
        for data_idx in self.data_idxs:
            emb_all.append(data_embs[data_idx])

        return np.concatenate(emb_all, axis=0)

    def _check_idxs(self, idx_list):
        if isinstance(idx_list[0], int):
            data_names = list(self.data_dict.keys())
            idx_list = [data_names[idx] for idx in idx_list]
        idx_list = sorted(list(set(idx_list)))

        return idx_list

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


class VideoSample(Dataset):
    """
    Dataset with sequential sampling
    """
    def __init__(self, data_dict, data_idxs, data_features, is_train=True, get_name=False):
        super(VideoSample, self).__init__()
        self.data_dict = data_dict
        self.data_features = data_features
        self.data_names = self._check_idxs(data_idxs)
        self.is_train = is_train
        self.get_name = get_name

    def __getitem__(self, idx):
        # Compose aug input
        data_name = self.data_names[idx]
        img_features = np.stack(self.data_features[data_name], axis=0)
        img_names = self.data_dict[data_name]["img"]

        if self.get_name:
            if self.is_train:
                labels = np.array(self.data_dict[data_name]["phase"])
                return img_features, labels, img_names
            else:
                return img_features, img_names
        else:
            if self.is_train:
                labels = np.array(self.data_dict[data_name]["phase"])
                return img_features, labels
            else:
                return img_features

    def __len__(self):

        return len(self.data_names)

    def _check_idxs(self, idx_list):
        if isinstance(idx_list[0], int):
            data_names = list(self.data_dict.keys())
            idx_list = [data_names[idx] for idx in idx_list]
        idx_list = sorted(list(set(idx_list)))

        return idx_list