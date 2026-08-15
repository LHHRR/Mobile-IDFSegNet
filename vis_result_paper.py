import argparse
import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sam import build_mobile_idfsegnet
from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode="bilinear", align_corners=False)
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description="Predict plantar masks with Mobile-IDFSegNet.")
    parser.add_argument("--model", "-m", required=True, metavar="FILE", help="Path to a Mobile-IDFSegNet .pth checkpoint.")
    parser.add_argument("--input", "-i", metavar="INPUT", nargs="+", required=True, help="Input image files or directories.")
    parser.add_argument("--output", "-o", metavar="OUTPUT", nargs="+", help="Optional output mask filenames.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/predictions"), help="Directory for generated masks and overlays.")
    parser.add_argument("--viz", "-v", action="store_true", help="Visualize predictions interactively.")
    parser.add_argument("--no-save", "-n", action="store_true", help="Do not save output masks.")
    parser.add_argument("--save-overlay", action="store_true", help="Save image overlays with predicted contours.")
    parser.add_argument("--mask-threshold", "-t", type=float, default=0.5, help="Binary mask threshold.")
    parser.add_argument("--scale", "-s", type=float, default=1.0, help="Scale factor for input images.")
    parser.add_argument("--classes", "-c", type=int, default=2, help="Number of classes.")
    parser.add_argument("--device", default=None, help="Device, e.g. cuda, cuda:0, or cpu.")
    return parser.parse_args()


def get_image_files(paths):
    image_files = []
    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    for path in paths:
        if os.path.isdir(path):
            for pattern in patterns:
                image_files.extend(glob.glob(os.path.join(path, pattern)))
        elif os.path.isfile(path):
            image_files.append(path)
    return sorted(image_files)


def output_name(input_file: str, args, index: int) -> Path:
    if args.output:
        if len(args.output) != len(args.input) and len(args.output) != len(get_image_files(args.input)):
            raise ValueError("--output must provide one filename per input image.")
        return Path(args.output[index])
    return args.output_dir / f"{Path(input_file).stem}_mask.png"


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, value in enumerate(mask_values):
        if value == 1 and mask_values == [0, 1]:
            out[mask == i] = 255
        else:
            out[mask == i] = value

    return Image.fromarray(out)


def draw_mask_boundaries(image, mask, color=(0, 255, 0), thickness=2):
    image_np = np.array(image)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_np, contours, -1, color, thickness=thickness)
    return Image.fromarray(image_np)


def load_checkpoint(model, checkpoint, device):
    state_dict = torch.load(checkpoint, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    mask_values = [0, 1]
    if isinstance(state_dict, dict) and "mask_values" in state_dict:
        state_dict = dict(state_dict)
        mask_values = state_dict.pop("mask_values")

    load_result = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded checkpoint %s (missing=%d, unexpected=%d)",
        checkpoint,
        len(load_result.missing_keys),
        len(load_result.unexpected_keys),
    )
    return mask_values


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    in_files = get_image_files(args.input)
    if not in_files:
        raise FileNotFoundError("No input images found.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net = build_mobile_idfsegnet(num_classes=args.classes, decoder="dbsd")
    mask_values = load_checkpoint(net, args.model, device)
    net.to(device=device)

    if not args.no_save:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, filename in enumerate(in_files):
        logging.info("Predicting image %s", filename)
        img = Image.open(filename).convert("RGB")
        mask = predict_img(
            net=net,
            full_img=img,
            scale_factor=args.scale,
            out_threshold=args.mask_threshold,
            device=device,
        )

        if not args.no_save:
            out_filename = output_name(filename, args, index)
            out_filename.parent.mkdir(parents=True, exist_ok=True)
            mask_to_image(mask, mask_values).save(out_filename)
            logging.info("Mask saved to %s", out_filename)

            if args.save_overlay:
                overlay_out_filename = out_filename.with_name(f"{out_filename.stem}_overlay.png")
                draw_mask_boundaries(img, mask).save(overlay_out_filename)
                logging.info("Overlay saved to %s", overlay_out_filename)

        if args.viz:
            plot_img_and_mask(img, mask)


if __name__ == "__main__":
    main()
