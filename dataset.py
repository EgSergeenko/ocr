from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CapchaDataset(Dataset):
    input_length_mapping = {3: 20, 4: 27, 5: 34}

    def __init__(
        self,
        seq_len: Union[int, Tuple[int, int]],
        img_h: int = 32,
        img_w: int = 28,
        split: str = 'digits',
        samples: int = None,
    ):
        self.emnist_dataset = datasets.EMNIST(
            './EMNIST', split=split, train=True, download=True,
        )
        self.seq_len = seq_len
        self.blank_label = len(self.emnist_dataset.classes)
        self.img_h = img_h
        self.img_w = img_w
        self.samples = samples
        self.num_classes = len(self.emnist_dataset.classes) + 1
        if isinstance(seq_len, int):
            self._min_seq_len = seq_len
            self._max_seq_len = seq_len
        elif (
                isinstance(seq_len, Tuple)
                and len(seq_len) == 2
                and isinstance(seq_len[0], int)
        ):
            self._min_seq_len = seq_len[0]
            self._max_seq_len = seq_len[1]

    def __len__(self):
        if self.samples is not None:
            return self.samples
        return len(self.emnist_dataset.classes) ** self._max_seq_len

    def __preprocess(self, random_images: torch.Tensor) -> np.ndarray:
        transformed_images = []
        for img in random_images:
            img = transforms.ToPILImage()(img)
            img = TF.rotate(img, -90, fill=[0.0])
            img = TF.hflip(img)
            img = TF.resize(img, size=(self.img_h, self.img_w))
            img = transforms.ToTensor()(img).numpy()
            transformed_images.append(img)
        images = np.array(transformed_images)
        images = np.hstack(
            images.reshape((len(transformed_images), self.img_h, self.img_w)),
        )
        full_img = np.zeros(shape=(self.img_h, self._max_seq_len * self.img_w)).astype(
            np.float32,
        )
        full_img[:, 0: images.shape[1]] = images
        return full_img

    def __getitem__(self, idx):
        random_seq_len = np.random.randint(
            self._min_seq_len,
            self._max_seq_len + 1,
        )
        random_indices = np.random.randint(
            len(self.emnist_dataset.data), size=(random_seq_len,),
        )
        random_images = self.emnist_dataset.data[random_indices]
        random_digits_labels = self.emnist_dataset.targets[random_indices]
        labels = torch.zeros((1, self._max_seq_len))
        labels = torch.fill(labels, self.blank_label)
        labels[0, 0: len(random_digits_labels)] = random_digits_labels
        x = np.expand_dims(self.__preprocess(random_images), axis=0)
        y = labels.numpy().reshape(self._max_seq_len)

        return x, y, self.input_length_mapping[random_seq_len], random_seq_len
