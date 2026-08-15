import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from evaluate_old import evaluate
from sam import build_mobile_idfsegnet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss


DEFAULT_IMAGES_DIR = Path("data/PT-3/Thermal_Image_PT3")
DEFAULT_MASKS_DIR = Path("data/PT-3/Thermal_Image_SegmentationMasks_PT3")
DEFAULT_CHECKPOINT_DIR = Path("outputs/checkpoints_dbsd")
DEFAULT_ENCODER_CHECKPOINT = Path("weights/mobile_sam.pt")


def configure_logging(checkpoint_dir: Path) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log_file_path = checkpoint_dir / "training_log.txt"
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)


def resolve_optional_checkpoint(path: Optional[str], required: bool = False) -> Optional[str]:
    if not path:
        return None

    checkpoint = Path(path)
    if checkpoint.is_file():
        return str(checkpoint)

    message = f"Checkpoint not found: {checkpoint}"
    if required:
        raise FileNotFoundError(message)

    logging.warning("%s. Continuing without it.", message)
    return None


def load_model_checkpoint(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    state_dict = torch.load(checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    if isinstance(state_dict, dict) and "mask_values" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("mask_values")

    load_result = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded training checkpoint %s (missing=%d, unexpected=%d)",
        checkpoint,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )


def build_dataset(
    images_dir: Path,
    masks_dir: Path,
    img_scale: float,
    mask_suffix: str = "",
    ids=None,
    mask_values=None,
    augment: bool = False,
) -> BasicDataset:
    return BasicDataset(
        images_dir,
        masks_dir,
        img_scale,
        mask_suffix=mask_suffix,
        ids=ids,
        mask_values=mask_values,
        augment=augment,
    )


def train_model(
    model,
    train_set,
    val_set,
    mask_values,
    device,
    checkpoint_dir: Path,
    epochs: int = 30,
    batch_size: int = 4,
    learning_rate: float = 5e-4,
    save_checkpoint: bool = True,
    img_scale: float = 1,
    amp: bool = False,
    weight_decay: float = 1e-8,
    gradient_clipping: float = 1.0,
    num_workers: Optional[int] = None,
):
    n_train = len(train_set)
    n_val = len(val_set)
    if num_workers is None:
        num_workers = os.cpu_count() or 0

    loader_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(
        "Starting training:\n"
        "  Epochs: %d\n"
        "  Batch size: %d\n"
        "  Learning rate: %.6g\n"
        "  Training size: %d\n"
        "  Validation size: %d\n"
        "  Checkpoints: %s\n"
        "  Device: %s\n"
        "  Image scale: %.3f\n"
        "  Mixed precision: %s",
        epochs,
        batch_size,
        learning_rate,
        n_train,
        n_val,
        save_checkpoint,
        device,
        img_scale,
        amp,
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=3)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                images, true_masks = batch["image"], batch["mask"]
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True,
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{"loss": loss.item()})

        val_score = evaluate(model, val_loader, device, amp)
        val_score_value = float(val_score.detach().cpu()) if torch.is_tensor(val_score) else float(val_score)
        scheduler.step(val_score_value)
        logging.info("Epoch %d validation Dice score: %.5f", epoch, val_score_value)

        if save_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["mask_values"] = mask_values
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_{val_score_value:.5f}.pth"
            torch.save(state_dict, checkpoint_path)
            logging.info("Checkpoint saved: %s", checkpoint_path)


def get_args():
    parser = argparse.ArgumentParser(description="Train Mobile-IDFSegNet on plantar thermograms.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Training image directory.")
    parser.add_argument("--masks-dir", type=Path, default=DEFAULT_MASKS_DIR, help="Training mask directory.")
    parser.add_argument("--mask-suffix", default="", help="Suffix appended to mask filenames before extension.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR, help="Output checkpoint directory.")
    parser.add_argument("--encoder-checkpoint", type=str, default=str(DEFAULT_ENCODER_CHECKPOINT), help="Optional MobileSAM encoder checkpoint.")
    parser.add_argument("--load", type=str, default=None, help="Optional full Mobile-IDFSegNet checkpoint to resume/fine-tune.")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs.")
    parser.add_argument("--batch-size", "-b", dest="batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--learning-rate", "-l", dest="lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Image downscaling factor.")
    parser.add_argument("--validation", "-v", dest="val", type=float, default=20.0, help="Validation split percentage.")
    parser.add_argument("--amp", action="store_true", default=False, help="Use mixed precision.")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/validation split.")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze the MobileSAM image encoder.")
    parser.add_argument("--no-augment", action="store_true", help="Disable training-time color augmentation.")
    parser.add_argument("--no-save-checkpoint", action="store_true", help="Disable checkpoint saving.")
    return parser.parse_args()


def main():
    args = get_args()
    configure_logging(args.checkpoint_dir)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Using device %s", device)

    base_dataset = build_dataset(args.images_dir, args.masks_dir, args.scale, mask_suffix=args.mask_suffix)
    ids = base_dataset.ids
    n_val = int(len(ids) * args.val / 100)
    n_train = len(ids) - n_val
    indices = torch.randperm(len(ids), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ids = [ids[i] for i in indices[:n_train]]
    val_ids = [ids[i] for i in indices[n_train:]]
    train_set = build_dataset(
        args.images_dir,
        args.masks_dir,
        args.scale,
        mask_suffix=args.mask_suffix,
        ids=train_ids,
        mask_values=base_dataset.mask_values,
        augment=not args.no_augment,
    )
    val_set = build_dataset(
        args.images_dir,
        args.masks_dir,
        args.scale,
        mask_suffix=args.mask_suffix,
        ids=val_ids,
        mask_values=base_dataset.mask_values,
        augment=False,
    )

    encoder_checkpoint = resolve_optional_checkpoint(args.encoder_checkpoint)
    model = build_mobile_idfsegnet(num_classes=args.classes, checkpoint=encoder_checkpoint, decoder="dbsd")

    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if name.startswith("image_encoder"):
                param.requires_grad = False

    if args.load:
        load_model_checkpoint(model, resolve_optional_checkpoint(args.load, required=True), device)

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
        checkpoint_dir=args.checkpoint_dir,
        img_scale=args.scale,
        amp=args.amp,
        save_checkpoint=not args.no_save_checkpoint,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
