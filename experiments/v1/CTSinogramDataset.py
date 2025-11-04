import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class CTSinogramDataset(Dataset):
    def __init__(self, image_dirs, sinogram_dirs, transform=None):
        assert len(image_dirs) == len(sinogram_dirs), "Mismatch in directories count"
        self.transform = transform
        self.pairs = []

        for img_dir, sino_dir in zip(image_dirs, sinogram_dirs):
            image_filenames = sorted(
                [f for f in os.listdir(img_dir) if f.endswith(".png")]
            )
            for img_name in image_filenames:
                sino_name = f"sinogram_{img_name}"
                img_path = os.path.join(img_dir, img_name)
                sino_path = os.path.join(sino_dir, sino_name)

                if os.path.exists(img_path) and os.path.exists(sino_path):
                    self.pairs.append((sino_path, img_path))
                else:
                    print(f"Skipping pair: {img_path}, {sino_path} (file missing)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sino_path, img_path = self.pairs[idx]

        image = Image.open(img_path).convert("L")
        sinogram = Image.open(sino_path).convert("L")

        image = np.array(image, dtype=np.float32) / 255.0
        sinogram = np.array(sinogram, dtype=np.float32) / 255.0

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        sinogram_tensor = torch.from_numpy(sinogram).unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            sinogram_tensor = self.transform(sinogram_tensor)

        return sinogram_tensor, image_tensor
