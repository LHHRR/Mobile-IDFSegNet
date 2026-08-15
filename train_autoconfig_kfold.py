import argparse
import logging
from pathlib import Path

import torch
from sklearn.model_selection import KFold

from sam import build_mobile_idfsegnet
from train_autoconfig import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_ENCODER_CHECKPOINT,
    DEFAULT_IMAGES_DIR,
    DEFAULT_MASKS_DIR,
    build_dataset,
    configure_logging,
    load_model_checkpoint,
    resolve_optional_checkpoint,
    train_model,
)


def get_args():
    parser = argparse.ArgumentParser(description="Train Mobile-IDFSegNet with k-fold validation.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Training image directory.")
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR, help="Training mask directory.")
    parser.add_argument("--mask-suffix", default="", help="Suffix appended to mask filenames before extension.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR.parent / "checkpoints_dbsd_kfold", help="Output checkpoint directory.")
    parser.add_argument("--encoder-checkpoint", type=str, default=str(DEFAULT_ENCODER_CHECKPOINT), help="Optional MobileSAM encoder checkpoint.")
    parser.add_argument("--load", type=str, default=None, help="Optional full Mobile-IDFSegNet checkpoint to resume/fine-tune.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--fold", default="0", help="Fold index to run, or 'all'.")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Image downscaling factor.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision.")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for k-fold split.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the MobileSAM image encoder.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time color augmentation.")
    parser.add_argument("--no-save-checkpoint", action="store_true", help="Disable checkpoint saving.")
    return parser.parse_args()


def selected_folds(fold: str, folds: int):
    if fold.lower() == "all":
        return list(range(folds))
    fold_index = int(fold)
    if fold_index < 0 or fold_index >= folds:
        raise ValueError(f"Fold index must be in [0, {folds - 1}], got {fold_index}.")
    return [fold_index]


def main():
    args = get_args()
    configure_logging(args.checkpoint_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device %s", device)

    base_dataset = build_dataset(args.images_dir, args.masks_dir, args.scale, mask_suffix=args.mask_suffix)
    ids = base_dataset.ids
    encoder_checkpoint = resolve_optional_checkpoint(args.encoder_checkpoint)
    training_checkpoint = resolve_optional_checkpoint(args.load, required=True) if args.load else None

    kfold = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(kfold.split(ids))

    for fold_index in selected_folds(args.fold, args.folds):
        logging.info("Starting fold %d/%d", fold_index, args.folds - 1)
        train_ids, val_ids = splits[fold_index]
        train_set = build_dataset(
            args.images_dir,
            args.masks_dir,
            args.scale,
            mask_suffix=args.mask_suffix,
            ids=[ids[i] for i in train_ids],
            mask_values=base_dataset.mask_values,
            augment=not args.no_augment,
        )
        val_set = build_dataset(
            args.images_dir,
            args.masks_dir,
            args.scale,
            mask_suffix=args.mask_suffix,
            ids=[ids[i] for i in val_ids],
            mask_values=base_dataset.mask_values,
            augment=False,
        )

        model = build_mobile_idfsegnet(num_classes=args.classes, checkpoint=encoder_checkpoint, decoder="dbsd")
        if args.freeze_encoder:
            for name, param in model.named_parameters():
                if name.startswith("image_encoder"):
                    param.requires_grad = False
        if training_checkpoint:
            load_model_checkpoint(model, training_checkpoint, device)

        model.to(device=device)
        train_model(
            model=model,
            train_set=train_set,
            val_set=val_set,
            mask_values=base_dataset.mask_values,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            checkpoint_dir=args.checkpoint_dir / f"fold_{fold_index}",
            img_scale=args.scale,
            amp=args.amp,
            save_checkpoint=not args.no_save_checkpoint,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
