import os
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms
from torchvision.transforms import Lambda

from utils.augment import RandomCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from dataset.esd import CholecDataset

def split_data(save_file, train_idxs, val_idxs, test_idxs):
    with open(save_file, 'rb') as f:
        all_info = pickle.load(f)

    train_file_paths = []
    test_file_paths = []

    val_file_paths = []
    val_labels = []

    train_labels = []
    test_labels = []

    train_num_each = []  # Number of samples in each video
    val_num_each = []
    test_num_each = []

    for i in train_idxs:
        train_num_each.append(len(all_info[i]))
        for idx in range(len(all_info[i])):
            train_file_paths.append(all_info[i][idx][0])
            train_labels.append(all_info[i][idx][1:])

    for i in val_idxs:
        val_num_each.append(len(all_info[i]))
        for idx in range(len(all_info[i])):
            val_file_paths.append(all_info[i][idx][0])
            val_labels.append(all_info[i][idx][1:])

    for i in test_idxs:
        test_num_each.append(len(all_info[i]))
        for idx in range(len(all_info[i])):
            test_file_paths.append(all_info[i][idx][0])
            test_labels.append(all_info[i][idx][1:])

    return [train_file_paths, train_labels, train_num_each], [val_file_paths, val_labels, val_num_each], [test_file_paths,
                                                                                                          test_labels,
                                                                                                          test_num_each]

def get_data(data_path, args, is_train=True):
    train_data, val_data, test_data = split_data(data_path,
                                                 train_idxs=list(range(10)),
                                                 val_idxs=list(range(10, 13)),
                                                 test_idxs=list(range(13, 17)))

    train_paths, train_labels, train_num_each = train_data
    val_paths, val_labels, val_num_each = val_data
    test_paths, test_labels, test_num_each = test_data

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    train_transforms = None
    test_transforms = None

    if args["flip"] == 0:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224, args["seq"]),
            RandomHorizontalFlip(args["seq"]),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif args["flip"] == 1:
        train_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            RandomCrop(224, args["seq"]),
            ColorJitter(args["seq"], brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            RandomHorizontalFlip(args["seq"]),
            RandomRotation(5, args["seq"]),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])

    if args["crop"] == 0:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif args["crop"] == 1:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif args["crop"] == 2:
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])
        ])
    elif args["crop"] == 5:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])
    elif args["crop"] == 10:
        test_transforms = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.41757566, 0.26098573, 0.25888634], [0.21938758, 0.1983, 0.19342837])(crop)
                     for crop in crops]))
        ])

    # train_dataset_19 = CholecDataset(train_paths_19, train_labels_19, train_transforms)
    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    if not is_train:
        train_dataset_LFB = CholecDataset(train_paths, train_labels, test_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    print("Finish get_data")
    if not is_train:
        return (train_dataset, train_dataset_LFB), train_num_each, val_dataset, val_num_each, test_dataset, test_num_each
    else:
        return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each

