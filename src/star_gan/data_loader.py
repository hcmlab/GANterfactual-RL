from typing import Tuple, Any

import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, DatasetFolder
from PIL import Image
import torch
import os
import random


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class PacmanNumpyDataset(DatasetFolder):
    """
    Custom dataset class for Pacman datasets with stacked frames saved as .npy files.
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            # transform each channel separately and concat afterwards to allow for more than 3 channels
            channel_samples = []
            for channel in range(sample.shape[-1]):
                channel_sample = np.asarray(sample[:, :, channel], dtype=np.float32)
                channel_sample = Image.fromarray(channel_sample, "F")
                channel_sample = self.transform(channel_sample)
                channel_samples.append(channel_sample)
            sample = torch.Tensor(len(channel_samples), *list(channel_samples[0].shape))
            torch.cat(channel_samples, out=sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class PacmanStackedImageDataset(DatasetFolder):
    """
    Custom dataset class for Pacman datasets with stacked frames saved as .npy files.
    """

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        stacked_frames = []
        stacked_frames_path_prefix = path[:-5]
        stacked_frames_path_suffix = path[-4:]
        for i in range(4):
            sample = Image.open(stacked_frames_path_prefix + f"{i}{stacked_frames_path_suffix}")
            if self.transform is not None:
                sample = self.transform(sample)
            stacked_frames.append(sample)

        stacked_frames = np.stack(stacked_frames)
        stacked_frames = stacked_frames.reshape((12, stacked_frames.shape[-2], stacked_frames.shape[-1]))

        return stacked_frames, target

    def __len__(self):
        """Return the number of images."""
        return int(np.ceil(len(self.samples) / 4))


def get_star_gan_transform(crop_size, image_size, image_channels):
    transform = []
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    if image_channels == 1 or image_channels == 4:
        transform.append(T.Normalize(mean=(0.5,), std=(0.5,)))
    else:
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    return transform


def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1, image_channels=3):
    """Build and return a data loader."""
    transform = get_star_gan_transform(crop_size, image_size, image_channels)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        # use custom Pacman dataset instead to allow for numpy array input with more than 3 channels
        if image_channels == 3:
            dataset = ImageFolder(image_dir, transform)
        elif image_channels == 12:
            dataset = PacmanStackedImageDataset(image_dir, None, extensions=(".png",), transform=transform)
        else:
            dataset = PacmanNumpyDataset(image_dir, np.load, extensions=(".npy",), transform=transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader
