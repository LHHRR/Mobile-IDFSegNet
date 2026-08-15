import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SamFeatSeg(nn.Module):
    def __init__(
        self,
        image_encoder,
        seg_decoder,
        img_size=1024,
        n_classes=None,
    ):
        super().__init__()
        self.img_size = img_size
        self.image_encoder = image_encoder
        self.mask_decoder = seg_decoder
        self.n_classes = n_classes

    def forward(self,
                x):
        original_size_h, original_size_w = x.shape[-2], x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        image_embedding = self.image_encoder(x) #[B, 256, 64, 64]
        out = self.mask_decoder(image_embedding)
        if out.shape[-1] != original_size_w or out.shape[-2] != original_size_h:
            out = F.interpolate(
                out,
                (original_size_h, original_size_w),
                mode="bilinear",
                align_corners=False,
            )
        return out

    def get_embedding(self, x):
        original_size = x.shape[-1]
        x = F.interpolate(
            x,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        image_embedding = self.image_encoder(x)
        out = nn.functional.adaptive_avg_pool2d(image_embedding, 1).squeeze()
        return out

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SegDecoderDBSD(nn.Module):
    """Dual-Branch Sampling Decoder (DBSD) used by Mobile-IDFSegNet."""

    def __init__(self,
                 num_classes=14,
                 emb_dim=256,
                 activation: Type[nn.Module] = nn.GELU,
                 ):
        super().__init__()

        self.output_upscaling_1 = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 4),
            activation(),
        )
        self.interpolate_conv_1 = nn.Sequential(
            conv1x1(emb_dim, emb_dim // 4),
            LayerNorm2d(emb_dim // 4),
            activation(),
        )

        self.output_upscaling_2 = nn.Sequential(nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 8),
            activation(),
        )
        self.interpolate_conv_2 = nn.Sequential(
            conv1x1(emb_dim // 4, emb_dim // 8),
            LayerNorm2d(emb_dim // 8),
            activation(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(emb_dim // 8, emb_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 8, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.output_upscaling_1(x)
        interpolate_out = self.interpolate_conv_1(x)
        interpolate = F.interpolate(interpolate_out, size=[out.shape[2],  out.shape[3]],
                                    mode='bilinear', align_corners=False) 
        x = out + interpolate

        out = self.output_upscaling_2(x)
        interpolate_out = self.interpolate_conv_2(x)
        interpolate = F.interpolate(interpolate_out, size=[out.shape[2],  out.shape[3]],
                                    mode='bilinear', align_corners=False) 
        x = out + interpolate

        return self.final(x)


class SegDecoderLinear(nn.Module):
    def __init__(self,
                 num_classes=14,
                 emb_dim=256,
                 activation: Type[nn.Module] = nn.GELU,
                 ):
        super().__init__()

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(emb_dim // 4),
            activation(),
            nn.ConvTranspose2d(emb_dim // 4, emb_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(emb_dim // 8, emb_dim // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(emb_dim // 8, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.output_upscaling(x)
        return self.final(out)
        

class SegDecoderCNN(nn.Module):
    def __init__(self,
                 num_classes=14,
                 embed_dim=256,
                 num_depth=2,
                 top_channel=64,
                 ):
        super().__init__()

        self.input_block = nn.Sequential(
            nn.Conv2d(embed_dim, top_channel*2**num_depth, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(top_channel * 2 ** num_depth, top_channel * 2 ** num_depth, kernel_size=1),
            nn.ReLU(inplace=True),

        )
        self.blocks = nn.ModuleList()
        for i in range(num_depth):
            if num_depth > 2 > i:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                block = nn.Sequential(
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(top_channel * 2 ** (num_depth - i),  top_channel * 2 ** (num_depth - i), 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(top_channel * 2 ** (num_depth - i), top_channel * 2 ** (num_depth - i - 1), 2,
                                       stride=2)
                )

            self.blocks.append(block)

        self.final = nn.Sequential(
            nn.Conv2d(top_channel, top_channel, 3, padding=1),
            nn.Conv2d(top_channel, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_block(x)
        for blk in self.blocks:
            x = blk(x)
        return self.final(x)

SegDecoderBUSD = SegDecoderDBSD




# from ptflops import get_model_complexity_info

# # Define input size
# input_size = (256, 64, 64)  # Example input size (channels, height, width)

# # Define the models
# linear_decoder = SegDecoderLinear()
# cnn_decoder = SegDecoderCNN()

# # Calculate complexity for Linear Decoder
# linear_flops, linear_params = get_model_complexity_info(linear_decoder, input_size, as_strings=True, print_per_layer_stat=False)
# print("Linear Decoder:")
# print("Flops:", linear_flops)
# print("Params:", linear_params)

# # Calculate complexity for CNN Decoder
# cnn_flops, cnn_params = get_model_complexity_info(cnn_decoder, input_size, as_strings=True, print_per_layer_stat=False)
# print("\nCNN Decoder:")
# print("Flops:", cnn_flops)
# print("Params:", cnn_params)
