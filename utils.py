import gc
import cv2
import torch
import numpy as np
from typing import Tuple
import torch.nn.utils.prune as prune

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clear_memory():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def calculate_padding(h: int, w: int, patch_size: int, step: int) -> Tuple[int, int, int, int]:
    patches_h = max(2, (h - patch_size + step) // step + 1)
    patches_w = max(2, (w - patch_size + step) // step + 1)
    padded_h = (patches_h - 1) * step + patch_size
    padded_w = (patches_w - 1) * step + patch_size
    return 0, padded_h - h, 0, padded_w - w


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    overlay = image.copy()

    overlay[mask == 1] = [0, 0, 255]   # Red
    overlay[mask == 2] = [0, 255, 0]   # Green

    blended = (
        image.astype(np.float32) * (1 - alpha)
        + overlay.astype(np.float32) * alpha
    ).astype(np.uint8)

    return blended


def filter_mask_by_class(mask: np.ndarray, choice: str):
    if choice == "Enamel":
        return (mask == 1).astype(np.uint8) * 1
    elif choice == "Pulp":
        return (mask == 2).astype(np.uint8) * 2
    else:  # Pulp + Enamel
        return mask



def apply_zoom(image: np.ndarray, zoom: float) -> np.ndarray:
    if zoom == 1.0:
        return image

    h, w = image.shape[:2]
    new_w, new_h = int(w * zoom), int(h * zoom)

    return cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )



def prune_conv_out_channels(conv, amount=0.3):
    prune.ln_structured(
        module=conv,
        name="weight",
        amount=amount,
        n=2,
        dim=0  # prune output channels
    )
    prune.remove(conv, "weight")


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def prune_u2net(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")
    return model