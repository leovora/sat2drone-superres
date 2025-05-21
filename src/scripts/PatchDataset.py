import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff


class PatchDataset(Dataset):
    def __init__(self, sentinel_paths, aerial_paths, patch_size=128, transform=True, normalize=True):
        assert len(sentinel_paths) == len(aerial_paths), "Liste di immagini disallineate."
        self.patch_size = patch_size
        self.transform = transform
        self.normalize = normalize

        self.images_info = []
        for s_path, a_path in zip(sentinel_paths, aerial_paths):
            sentinel = tiff.imread(s_path)
            aerial = tiff.imread(a_path)

            # Trasponi per ottenere (C, H, W)
            if sentinel.ndim == 3:
                sentinel = np.transpose(sentinel, (2, 0, 1))
            elif sentinel.ndim == 2:
                sentinel = sentinel[np.newaxis, :, :]
            else:
                continue  # skip formato non valido

            if aerial.ndim == 3:
                aerial = np.transpose(aerial, (2, 0, 1))
            elif aerial.ndim == 2:
                aerial = aerial[np.newaxis, :, :]
            else:
                continue  # skip formato non valido

            # Check dimensioni minime
            _, h_s, w_s = sentinel.shape
            _, h_a, w_a = aerial.shape
            h, w = min(h_s, h_a), min(w_s, w_a)

            if h >= patch_size and w >= patch_size:
                self.images_info.append({
                    "sentinel_path": s_path,
                    "aerial_path": a_path,
                    "shape": (h, w)
                })

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        for _ in range(10):  # massimo 10 tentativi
            info = self.images_info[idx]
            sentinel = tiff.imread(info["sentinel_path"]).astype(np.float32)
            aerial = tiff.imread(info["aerial_path"]).astype(np.float32)

            # (C, H, W)
            if sentinel.ndim == 3:
                sentinel = np.transpose(sentinel, (2, 0, 1))
            elif sentinel.ndim == 2:
                sentinel = sentinel[np.newaxis, :, :]

            if aerial.ndim == 3:
                aerial = np.transpose(aerial, (2, 0, 1))
            elif aerial.ndim == 2:
                aerial = aerial[np.newaxis, :, :]

            ps = self.patch_size
            h, w = info["shape"]

            # Coordinate valide
            if h - ps <= 0 or w - ps <= 0:
                continue

            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)

            sentinel_patch = sentinel[:, y:y+ps, x:x+ps]
            aerial_patch = aerial[:, y:y+ps, x:x+ps]

            # Skip se shape errata
            if sentinel_patch.shape[1:] != (ps, ps) or aerial_patch.shape[1:] != (ps, ps):
                continue

            # Skip se contiene NaN o inf
            if not np.isfinite(sentinel_patch).all() or not np.isfinite(aerial_patch).all():
                continue

            # Normalizzazione
            if self.normalize:
                sentinel_patch = (sentinel_patch - sentinel_patch.mean()) / (sentinel_patch.std() + 1e-8)
                aerial_patch = (aerial_patch - aerial_patch.mean()) / (aerial_patch.std() + 1e-8)

            # Augmentation
            if self.transform:
                if random.random() > 0.5:
                    sentinel_patch = np.flip(sentinel_patch, axis=2)
                    aerial_patch = np.flip(aerial_patch, axis=2)
                if random.random() > 0.5:
                    sentinel_patch = np.flip(sentinel_patch, axis=1)
                    aerial_patch = np.flip(aerial_patch, axis=1)
                if random.random() > 0.5:
                    sentinel_patch = np.rot90(sentinel_patch, k=1, axes=(1, 2))
                    aerial_patch = np.rot90(aerial_patch, k=1, axes=(1, 2))

            return torch.from_numpy(sentinel_patch.copy()), torch.from_numpy(aerial_patch.copy())

        raise RuntimeError(f"Impossibile estrarre una patch valida per idx {idx}")



def create_dataloader(sentinel_paths, aerial_paths, batch_size=16, shuffle=True, num_workers=4):
    dataset = PatchDataset(sentinel_paths, aerial_paths,transform=True, normalize=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)