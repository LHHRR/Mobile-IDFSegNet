import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from evaluate_old import evaluate
from sam import build_mobile_idfsegnet
from train_autoconfig import DEFAULT_IMAGES_DIR, DEFAULT_MASKS_DIR, load_model_checkpoint
from utils.data_loading import BasicDataset


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate a Mobile-IDFSegNet checkpoint.")
    parser.add_argument("--model", "-m", required=True, help="Path to a Mobile-IDFSegNet .pth checkpoint.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Evaluation image directory.")
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR, help="Evaluation mask directory.")
    parser.add_argument("--mask-suffix", default="", help="Suffix appended to mask filenames before extension.")
    parser.add_argument("--validation", "-v", dest="val", type=float, default=20.0, help="Validation split percentage.")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Image downscaling factor.")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for validation split.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision.")
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device %s", device)

    dataset = BasicDataset(args.images_dir, args.masks_dir, args.scale, mask_suffix=args.mask_suffix)
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else (os.cpu_count() or 0),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    model = build_mobile_idfsegnet(num_classes=args.classes, decoder="dbsd")
    load_model_checkpoint(model, args.model, device)
    model.to(device=device)

    val_score = evaluate(model, val_loader, device, args.amp)
    val_score_value = float(val_score.detach().cpu()) if torch.is_tensor(val_score) else float(val_score)
    logging.info("Validation Dice score: %.5f", val_score_value)
    print(f"{val_score_value:.5f}")


if __name__ == "__main__":
    main()
