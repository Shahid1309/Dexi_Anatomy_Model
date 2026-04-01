# import cv2
# import torch
# import numpy as np
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from patchify import patchify, unpatchify
# from model import U2NETMultiClass
# from utils import calculate_padding, clear_memory
# import warnings

# warnings.filterwarnings("ignore")

# # =====================================================
# # CONFIG
# # =====================================================
# MODEL_PATH = "best_iou_model.pth"
# NUM_CLASSES = 3
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # =====================================================
# # MODEL
# # =====================================================

# def load_model():
#     model = torch.jit.load("u2net_scripted.pt", map_location=DEVICE)
#     model.eval()

#     if DEVICE.type == "cuda":
#         torch.backends.cudnn.benchmark = True

#     return model

# def pad_tensor_to_multiple(x: torch.Tensor, multiple: int = 32):
#     _, _, h, w = x.shape
#     pad_h = (multiple - h % multiple) % multiple
#     pad_w = (multiple - w % multiple) % multiple
#     return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h)), h, w



# # =====================================================
# # TRANSFORM
# # =====================================================
# transform = A.Compose([
#     A.Normalize(mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)),
#     ToTensorV2()
# ])


# # @torch.no_grad()
# # def predict_patch(patch: np.ndarray, model):
# #     augmented = transform(image=patch)
# #     x = augmented["image"].unsqueeze(0).to(DEVICE)

# #     if DEVICE.type == "cuda":
# #         x = x.half()

# #     with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
# #         out = model(x)

# #     pred = out[0].argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

# #     del x, out
# #     clear_memory()

# #     return pred
# @torch.no_grad()
# def predict_patch(patch: np.ndarray, model):
#     augmented = transform(image=patch)
#     x = augmented["image"].unsqueeze(0).to(DEVICE)

#     if DEVICE.type == "cuda":
#         x = x.half()

#     # ✅ PAD TO MULTIPLE OF 32 (CRITICAL)
#     x, orig_h, orig_w = pad_tensor_to_multiple(x, 32)

#     with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
#         out = model(x)

#     # Take main output, argmax
#     pred = out[0].argmax(dim=1)

#     # ✅ CROP BACK TO ORIGINAL PATCH SIZE
#     pred = pred[:, :orig_h, :orig_w]

#     pred = pred.squeeze().cpu().numpy().astype(np.uint8)

#     del x, out
#     clear_memory()

#     return pred



# def predict_large_image(  image_rgb: np.ndarray, model, patch_size: int = 1024, step: int = 128):

#     h, w = image_rgb.shape[:2]

#     pad_t, pad_b, pad_l, pad_r = calculate_padding(h, w, patch_size, step)
#     image_padded = cv2.copyMakeBorder(
#         image_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT
#     )

#     patches = patchify(image_padded, (patch_size, patch_size, 3), step=step)
#     ph, pw = patches.shape[0], patches.shape[1]

#     pred_patches = np.zeros((ph, pw, patch_size, patch_size), dtype=np.uint8)

#     for i in range(ph):
#         for j in range(pw):
#             pred_patches[i, j] = predict_patch(patches[i, j, 0], model)

#             if (i * pw + j) % 10 == 0:
#                 clear_memory()

#     pred_mask_padded = unpatchify(pred_patches, image_padded.shape[:2])
#     pred_mask = pred_mask_padded[:h, :w]
#     print("Unique labels:", np.unique(pred_mask))

#     del patches, pred_patches, pred_mask_padded
#     clear_memory()

#     return pred_mask



import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from patchify import patchify, unpatchify
from model import U2NETMultiClass
from utils import calculate_padding, clear_memory
import warnings
warnings.filterwarnings("ignore")
# =====================================================
# CONFIG
# =====================================================
MODEL_PATH = "best_iou_model.pth"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =====================================================
# MODEL
# =====================================================
def load_model():
    model = U2NETMultiClass(n_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return model

def pad_tensor_to_multiple(x: torch.Tensor, multiple: int = 32):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h)), h, w

# =====================================================
# TRANSFORM
# =====================================================
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@torch.no_grad()
def predict_patch(patch: np.ndarray, model):
    augmented = transform(image=patch)
    x = augmented["image"].unsqueeze(0).to(DEVICE)
    if DEVICE.type == "cuda":
        x = x.half()
    # ✅ PAD TO MULTIPLE OF 32 (CRITICAL)
    x, orig_h, orig_w = pad_tensor_to_multiple(x, 32)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        out = model(x)
    # Take main output, argmax
    pred = out[0].argmax(dim=1)
    # ✅ CROP BACK TO ORIGINAL PATCH SIZE
    pred = pred[:, :orig_h, :orig_w]
    pred = pred.squeeze().cpu().numpy().astype(np.uint8)
    del x, out
    clear_memory()
    return pred

def predict_large_image(
    image_rgb: np.ndarray,
    model,
    patch_size: int = 1024,
    step: int = 128,
):
    h, w = image_rgb.shape[:2]
    pad_t, pad_b, pad_l, pad_r = calculate_padding(h, w, patch_size, step)
    image_padded = cv2.copyMakeBorder(
        image_rgb, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT
    )
    patches = patchify(image_padded, (patch_size, patch_size, 3), step=step)
    ph, pw = patches.shape[0], patches.shape[1]
    pred_patches = np.zeros((ph, pw, patch_size, patch_size), dtype=np.uint8)
    for i in range(ph):
        for j in range(pw):
            pred_patches[i, j] = predict_patch(patches[i, j, 0], model)
            if (i * pw + j) % 10 == 0:
                clear_memory()
    pred_mask_padded = unpatchify(pred_patches, image_padded.shape[:2])
    pred_mask = pred_mask_padded[:h, :w]
    print("Unique labels:", np.unique(pred_mask))
    del patches, pred_patches, pred_mask_padded
    clear_memory()
    return pred_mask