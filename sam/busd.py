"""Backward-compatible decoder import.

The paper and public code use Dual-Branch Sampling Decoder (DBSD).  Older
experiments used the BUSD name, so this module keeps that import path working.
"""

from .SamFeatSeg import SegDecoderDBSD


BUSD = SegDecoderDBSD
