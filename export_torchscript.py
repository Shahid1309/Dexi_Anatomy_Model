import torch
from model import U2NETMultiClass

MODEL_PATH = "best_iou_model.pth"
SCRIPTED_PATH = "u2net_scripted.pt"
NUM_CLASSES = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = U2NETMultiClass(n_classes=NUM_CLASSES).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint.get("model_state_dict", checkpoint)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)

model.eval()

# 🔥 FP16 for speed (optional but recommended)
if DEVICE.type == "cuda":
    model = model.half()
    example = torch.randn(1, 3, 1024, 1024, device=DEVICE).half()
else:
    example = torch.randn(1, 3, 1024, 1024)

scripted = torch.jit.trace(model, example)
scripted.save(SCRIPTED_PATH)

print("✅ TorchScript model saved:", SCRIPTED_PATH)
