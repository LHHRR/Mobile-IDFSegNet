import argparse
from time import perf_counter

import torch

try:
    from .build_sam import build_mobile_idfsegnet
except ImportError:
    from build_sam import build_mobile_idfsegnet


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark Mobile-IDFSegNet throughput.")
    parser.add_argument("--classes", type=int, default=2, help="Number of segmentation classes.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint path.")
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--image-size", type=int, default=1024, help="Square benchmark input size.")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up iterations.")
    parser.add_argument("--runs", type=int, default=1000, help="Measured iterations.")
    parser.add_argument("--flops", action="store_true", help="Also estimate FLOPs with ptflops if installed.")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_mobile_idfsegnet(num_classes=args.classes, checkpoint=args.checkpoint, decoder="dbsd")
    model.to(device=device).eval()

    if args.flops:
        try:
            from ptflops import get_model_complexity_info

            flops, params = get_model_complexity_info(
                model,
                (3, args.image_size, args.image_size),
                as_strings=True,
                print_per_layer_stat=False,
            )
            print(f"FLOPs: {flops}")
            print(f"Params: {params}")
        except ImportError:
            print("ptflops is not installed; skipping FLOPs estimation.")

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size, device=device)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(args.runs):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start

    print(f"Processed {args.runs} frames in {elapsed:.2f} seconds, FPS: {args.runs / elapsed:.2f}")


if __name__ == "__main__":
    main()
