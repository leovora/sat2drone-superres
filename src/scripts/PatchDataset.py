import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import tifffile as tiff


class PatchDataset(Dataset):
    def __init__(self, sentinel_paths, aerial_paths, patch_size=128, transform=True, normalize=True):
        """
        :param sentinel_paths: lista di path alle immagini Sentinel
        :param aerial_paths: lista di path alle immagini aeree (stessa lunghezza)
        :param patch_size: dimensione del patch quadrato (default 128x128)
        :param transform: se True, applica data augmentation
        :param normalize: se True, normalizza i valori dei pixel
        """
        assert len(sentinel_paths) == len(aerial_paths), "Liste di immagini disallineate."
        self.sentinel_paths = sentinel_paths
        self.aerial_paths = aerial_paths
        self.patch_size = patch_size
        self.transform = transform
        self.normalize = normalize

        # Carica metadati sulle dimensioni
        self.images_info = []
        for s_path, a_path in zip(self.sentinel_paths, self.aerial_paths):
            sentinel = tiff.imread(s_path)
            h, w = sentinel.shape[1:] if sentinel.ndim == 3 else sentinel.shape
            self.images_info.append({
                "sentinel_path": s_path,
                "aerial_path": a_path,
                "shape": (h, w)
            })

    def __len__(self):
        return len(self.sentinel_paths)

    def __getitem__(self, idx):
        # Scegli random una coppia di immagini
        info = random.choice(self.images_info)
        sentinel = tiff.imread(info["sentinel_path"]).astype(np.float32)
        aerial = tiff.imread(info["aerial_path"]).astype(np.float32)

        # Gestione canali singoli
        if sentinel.ndim == 2: sentinel = np.expand_dims(sentinel, axis=0)
        if aerial.ndim == 2: aerial = np.expand_dims(aerial, axis=0)

        h, w = info["shape"]
        ps = self.patch_size

        # Trova una posizione valida (evita bordi) -- TODO: trovare alternativa migliore (padding?)
        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        # Estrai patch
        sentinel_patch = sentinel[:, y:y+ps, x:x+ps]
        aerial_patch = aerial[:, y:y+ps, x:x+ps]

        # Gestione valori NaN o infiniti
        if np.isnan(sentinel_patch).any() or np.isnan(aerial_patch).any():
            return self.__getitem__(idx)  # Skippa patch non valide

        if self.normalize:
            sentinel_patch = (sentinel_patch - np.mean(sentinel_patch)) / (np.std(sentinel_patch) + 1e-8)
            aerial_patch = (aerial_patch - np.mean(aerial_patch)) / (np.std(aerial_patch) + 1e-8)

        # Data augmentation
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

        # Converti in tensor
        return torch.from_numpy(sentinel_patch), torch.from_numpy(aerial_patch)


# Uso tipico del DataLoader
def create_dataloader(sentinel_paths, aerial_paths, batch_size=16, shuffle=True, num_workers=4):
    dataset = PatchDataset(sentinel_paths, aerial_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)