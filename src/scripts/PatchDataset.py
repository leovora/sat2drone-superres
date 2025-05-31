import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tifffile as tiff
import random

class PatchGridDataset(Dataset):
    def __init__(self, sentinel_paths, aerial_paths, patch_size=128, stride=64, augment=True, normalize=True, max_patches_per_image=None):
        assert len(sentinel_paths) == len(aerial_paths), "Liste di immagini disallineate."

        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        self.normalize = normalize
        self.patches = []  # lista di (sentinel_patch, aerial_patch)

        for s_path, a_path in zip(sentinel_paths, aerial_paths):
            sentinel = tiff.imread(s_path).astype(np.float32)
            aerial = tiff.imread(a_path).astype(np.float32)

            sentinel = self._prepare_image(sentinel)
            aerial = self._prepare_image(aerial)

            # Crop al minimo comune shape
            h = min(sentinel.shape[1], aerial.shape[1])
            w = min(sentinel.shape[2], aerial.shape[2])
            sentinel = sentinel[:, :h, :w]
            aerial = aerial[:, :h, :w]

            # Estrai patch a griglia
            patches = self._extract_patches(sentinel, aerial)
            if max_patches_per_image:
                random.shuffle(patches)
                patches = patches[:max_patches_per_image]
            self.patches.extend(patches)

    def _prepare_image(self, img):
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        elif img.ndim == 3:
            img = np.transpose(img, (2, 0, 1))
        return img

    def _extract_patches(self, sentinel, aerial):
        ps = self.patch_size
        st = self.stride
        patches = []
        C, H, W = sentinel.shape

        for y in range(0, H - ps + 1, st):
            for x in range(0, W - ps + 1, st):
                sp = sentinel[:, y:y + ps, x:x + ps]
                ap = aerial[:, y:y + ps, x:x + ps]

                if not np.isfinite(sp).all() or not np.isfinite(ap).all():
                    continue
                valid_ratio = np.count_nonzero(sp) / sp.size
                if valid_ratio < 0.1:
                    continue

                if self.normalize:
                    sp = (sp - sp.mean()) / (sp.std() + 1e-8)
                    ap = (ap - ap.mean()) / (ap.std() + 1e-8)

                patches.append((sp, ap))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        sentinel, aerial = self.patches[idx]

        if self.augment:
            if random.random() < 0.5:
                sentinel = np.flip(sentinel, axis=1)
                aerial = np.flip(aerial, axis=1)
            if random.random() < 0.5:
                sentinel = np.flip(sentinel, axis=2)
                aerial = np.flip(aerial, axis=2)
            if random.random() < 0.5:
                sentinel = np.rot90(sentinel, k=1, axes=(1, 2))
                aerial = np.rot90(aerial, k=1, axes=(1, 2))
            if random.random() < 0.3:
                noise = np.random.normal(0, 0.05, size=sentinel.shape)
                sentinel = sentinel + noise
                aerial = aerial + noise

        return torch.from_numpy(sentinel.copy()), torch.from_numpy(aerial.copy())


def create_dataloaders(sentinel_paths, aerial_paths, batch_size=32, val_ratio=0.1, test_ratio=0.1, **kwargs):
    dataset = PatchGridDataset(sentinel_paths, aerial_paths, **kwargs)
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    n_train = n - n_val - n_test
    
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader