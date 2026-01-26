import itertools

import numpy as np
import torch
import torch.utils.data as Data

from utils.Functions import load_4D, imgnorm


class BaseDataset(Data.Dataset):
    """Base dataset class with common functionality."""

    def __init__(self, norm=True):
        super().__init__()
        self.norm = norm

    def _load_and_normalize(self, path):
        """Load image and optionally normalize."""
        img = load_4D(path)
        if self.norm:
            return torch.from_numpy(imgnorm(img)).float()
        return torch.from_numpy(img).float()

    def _load_pair(self, path_a, path_b):
        """Load and return a pair of images."""
        return self._load_and_normalize(path_a), self._load_and_normalize(path_b)


class Dataset(BaseDataset):
    """Dataset with random pair sampling for each iteration."""

    def __init__(self, names, iterations, norm=False):
        super().__init__(norm)
        self.names = names
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, step):
        index_pair = np.random.permutation(len(self.names))[:2]
        return self._load_pair(self.names[index_pair[0]], self.names[index_pair[1]])


class Dataset_epoch(BaseDataset):
    """Dataset that returns single images per epoch."""

    def __init__(self, names, norm=True):
        super().__init__(norm)
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, step):
        return self._load_and_normalize(self.names[step])


class Dataset_epoch_cam(BaseDataset):
    """Dataset with all permutation pairs from a single image list."""

    def __init__(self, names, norm=True):
        super().__init__(norm)
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        return self._load_pair(self.index_pair[step][0], self.index_pair[step][1])


class Dataset_epoch_mul(BaseDataset):
    """Dataset with cartesian product pairs from two image lists."""

    def __init__(self, names_1, names_2, norm=True):
        super().__init__(norm)
        self.index_pair = list(itertools.product(names_1, names_2))

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        return self._load_pair(self.index_pair[step][0], self.index_pair[step][1])


class Dataset_epoch_IXI(BaseDataset):
    """Dataset for IXI with cross-modality pairs (T1-T2, T1-PD, T2-PD)."""

    def __init__(self, name_t1, name_t2, name_pd, norm=True):
        super().__init__(norm)
        self.index_pair = (
            list(itertools.product(name_t1, name_t2)) +
            list(itertools.product(name_t1, name_pd)) +
            list(itertools.product(name_t2, name_pd))
        )

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        return self._load_pair(self.index_pair[step][0], self.index_pair[step][1])


class Dataset_Mapping(BaseDataset):
    """Dataset with pre-defined index pairs."""

    def __init__(self, index_pair, norm=True):
        super().__init__(norm)
        self.index_pair = index_pair

    def __len__(self):
        return len(self.index_pair)

    def __getitem__(self, step):
        return self._load_pair(self.index_pair[step][0], self.index_pair[step][1])


class Dataset_Mapping_val(BaseDataset):
    """Validation dataset with image pairs and corresponding label pairs."""

    def __init__(self, image_pair, label_pair, norm=True):
        super().__init__(norm)
        self.image_pair = image_pair
        self.label_pair = label_pair

    def __len__(self):
        return len(self.image_pair)

    def __getitem__(self, step):
        img_a, img_b = self._load_pair(self.image_pair[step][0], self.image_pair[step][1])
        label_a = torch.from_numpy(load_4D(self.label_pair[step][0])).float()
        label_b = torch.from_numpy(load_4D(self.label_pair[step][1])).float()
        return img_a, img_b, label_a, label_b


class Dataset_epoch_validation(BaseDataset):
    """Validation dataset with cross-product pairs of images and labels."""

    def __init__(self, name_1, name_2, label, norm=False):
        super().__init__(norm)
        self.imgs_pair = list(itertools.product(name_1, name_2))
        self.labels_pair = list(itertools.product(label, label))

    def __len__(self):
        return len(self.imgs_pair)

    def __getitem__(self, step):
        img_a, img_b = self._load_pair(self.imgs_pair[step][0], self.imgs_pair[step][1])
        label_a = torch.from_numpy(load_4D(self.labels_pair[step][0])).float()
        label_b = torch.from_numpy(load_4D(self.labels_pair[step][1])).float()
        return img_a, img_b, label_a, label_b
