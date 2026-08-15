import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff


def _hausdorff_distance(true_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    true_points = np.argwhere(true_mask)
    pred_points = np.argwhere(pred_mask)
    if true_points.size == 0 or pred_points.size == 0:
        return np.nan
    return max(
        directed_hausdorff(true_points, pred_points)[0],
        directed_hausdorff(pred_points, true_points)[0],
    )


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, foreground_class: int = 1):
    """Evaluate Dice, Hausdorff distance, pixel accuracy, and mean IoU."""
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    hausdorff_scores = []
    global_correct_total = 0
    iou_total = 0

    with torch.autocast(device.type if device.type != "mps" else "cpu", enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc="Validation round", unit="batch", leave=False):
            image, mask_true = batch["image"], batch["mask"]

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, "True mask indices should be in [0, n_classes["
            mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred_labels = mask_pred.argmax(dim=1)
            mask_pred_one_hot = F.one_hot(mask_pred_labels, net.n_classes).permute(0, 3, 1, 2).float()

            dice_score += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)

            mask_true_np = mask_true.cpu().numpy()
            mask_pred_np = mask_pred_labels.cpu().numpy()

            global_correct_total += np.mean(mask_true_np == mask_pred_np)
            iou_total += jaccard_score(
                mask_true_np.flatten(),
                mask_pred_np.flatten(),
                average="macro",
                zero_division=0,
            )

            for i in range(mask_true_np.shape[0]):
                hausdorff_scores.append(
                    _hausdorff_distance(
                        mask_true_np[i] == foreground_class,
                        mask_pred_np[i] == foreground_class,
                    )
                )

    avg_dice = dice_score / max(num_val_batches, 1)
    avg_hausdorff = float(np.nanmean(hausdorff_scores)) if hausdorff_scores else np.nan
    avg_global_correct = global_correct_total / max(num_val_batches, 1)
    avg_iou = iou_total / max(num_val_batches, 1)

    net.train()
    return {
        "Dice": avg_dice,
        "Hausdorff Distance": avg_hausdorff,
        "Global Correct": avg_global_correct,
        "Mean IoU": avg_iou,
    }
