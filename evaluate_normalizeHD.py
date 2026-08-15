import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import jaccard_score
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    global_correct_total = 0
    iou_total = 0
    hausdorff_distance_total = 0
    
    # Function to convert mask to point set
    def mask_to_point_set(mask):
        indices = np.column_stack(np.where(mask))
        # Normalize the points to [0, 1]
        indices = indices / np.array(mask.shape)
        return indices

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
            mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
            mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

            dice_score += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)

            mask_true_np = mask_true.cpu().numpy()
            mask_pred_np = mask_pred.argmax(dim=1).cpu().numpy()

            global_correct_total += np.sum(mask_true_np == mask_pred_np) / mask_true_np.size

            iou_total += jaccard_score(
                mask_true_np.flatten(), 
                mask_pred_np.flatten(),
                average='macro'
            )

            for i in range(net.n_classes):
                for image_index in range(mask_true_np.shape[0]):  # Loop through each image in the batch
                    mask_true_class = mask_true_np[image_index] == i
                    mask_pred_class = mask_pred_np[image_index] == i
                    if mask_true_class.any() and mask_pred_class.any():  # Ensure both classes are present
                        true_points = mask_to_point_set(mask_true_class)
                        pred_points = mask_to_point_set(mask_pred_class)
                        hausdorff_distance_total += directed_hausdorff(true_points, pred_points)[0]
            
    avg_dice = dice_score / max(num_val_batches, 1)
    avg_global_correct = global_correct_total / max(num_val_batches, 1)
    avg_iou = iou_total / max(num_val_batches, 1)
    avg_hausdorff_distance = hausdorff_distance_total / (num_val_batches * net.n_classes)

    net.train()
    return {
        'Dice': avg_dice,
        'Global Correct': avg_global_correct,
        'Mean IoU': avg_iou,
        'Mean Hausdorff Distance': avg_hausdorff_distance
    }
