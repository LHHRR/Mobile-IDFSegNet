import torch
from typing import Optional

from .SamFeatSeg import SamFeatSeg, SegDecoderCNN, SegDecoderDBSD, SegDecoderLinear
from .mobile_sam import TinyViT


def _make_decoder(decoder: str, num_classes: int):
    decoder = decoder.lower()
    if decoder == "dbsd":
        return SegDecoderDBSD(num_classes=num_classes)
    if decoder == "linear":
        return SegDecoderLinear(num_classes=num_classes)
    if decoder == "cnn":
        return SegDecoderCNN(num_classes=num_classes, num_depth=4)
    raise ValueError(f"Unsupported decoder '{decoder}'. Choose from: dbsd, linear, cnn.")


def _load_checkpoint(model: SamFeatSeg, checkpoint: str) -> None:
    state_dict = torch.load(checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    load_result = model.load_state_dict(state_dict, strict=False)
    missing = len(load_result.missing_keys)
    unexpected = len(load_result.unexpected_keys)
    print(f"Loaded checkpoint: {checkpoint} (missing={missing}, unexpected={unexpected})")


def build_mobile_idfsegnet(num_classes: int = 2, checkpoint: Optional[str] = None, decoder: str = "dbsd"):
    """Build Mobile-IDFSegNet with the decoder terminology used in the paper."""
    model = SamFeatSeg(
        image_encoder=TinyViT(
            img_size=1024,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        seg_decoder=_make_decoder(decoder, num_classes),
        n_classes=num_classes,
    )

    if checkpoint:
        _load_checkpoint(model, checkpoint)

    return model


def build_sam_mobile_seg_cnn(num_classes: int = 2, checkpoint: Optional[str] = None, decoder: str = "dbsd"):
    """Backward-compatible alias for older scripts."""
    return build_mobile_idfsegnet(num_classes=num_classes, checkpoint=checkpoint, decoder=decoder)
