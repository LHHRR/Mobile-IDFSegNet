import logging
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import isfile, join, splitext
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1].lower()
    if ext == ".npy":
        return Image.fromarray(np.load(filename))
    if ext in [".pt", ".pth"]:
        tensor = torch.load(filename, map_location="cpu")
        return Image.fromarray(tensor.numpy())
    return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + ".*"))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    if mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    raise ValueError(f"Loaded masks should have 2 or 3 dimensions, found {mask.ndim}")


class BasicDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        mask_dir: str,
        scale: float = 1.0,
        mask_suffix: str = "",
        ids=None,
        mask_values=None,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        self.mask_suffix = mask_suffix

        all_ids = [
            splitext(file)[0]
            for file in listdir(images_dir)
            if isfile(join(images_dir, file)) and not file.startswith(".")
        ]
        self.ids = list(ids) if ids is not None else all_ids
        if not self.ids:
            raise RuntimeError(f"No input file found in {images_dir}, make sure you put your images there")

        self.augmentations = (
            A.Compose(
                [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                ]
            )
            if augment
            else None
        )

        logging.info("Creating dataset with %d examples", len(self.ids))
        if mask_values is None:
            logging.info("Scanning mask files to determine unique values")
            with Pool() as p:
                unique = list(
                    tqdm(
                        p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                        total=len(self.ids),
                    )
                )
            self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        else:
            self.mask_values = mask_values
        logging.info("Unique mask values: %s", self.mask_values)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = int(scale * w), int(scale * h)
        assert new_w > 0 and new_h > 0, "Scale is too small, resized images would have no pixel"
        pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((new_h, new_w), dtype=np.int64)
            for i, value in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == value] = i
                else:
                    mask[(img == value).all(-1)] = i
            return mask

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))

        assert len(img_file) == 1 and len(mask_file) == 1, f"Issues with finding files for ID {name}"
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, f"Image and mask {name} size mismatch"

        if self.augmentations is not None:
            augmented = self.augmentations(image=np.array(img), mask=np.array(mask))
            img = Image.fromarray(augmented["image"])
            mask = Image.fromarray(augmented["mask"])

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            "image": torch.as_tensor(img.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix="_mask")
