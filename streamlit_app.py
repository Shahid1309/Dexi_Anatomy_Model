import cv2
import numpy as np
import streamlit as st
from infer import load_model, predict_large_image
from utils import create_overlay, filter_mask_by_class, apply_zoom

def apply_zoom(image, zoom_factor):
    h, w = image.shape[:2]
    new_h, new_w = int(h*zoom_factor), int(w*zoom_factor)
    zoomed_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return zoomed_img


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="🦷 Dental X-ray Segmentation", layout="wide")

st.markdown("<h1 style='text-align:center'>🦷 Dental X-ray Segmentation</h1>", unsafe_allow_html=True)
st.caption("Patch-based semantic segmentation for dental radiographs")

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
body { background-color: #f0f2f6; font-family: 'Inter', sans-serif;}
h1 { color: #0f172a; font-weight: 700;}
.stButton>button { background: linear-gradient(135deg,#1f77b4,#155a8a); color:white; border-radius:12px; height:46px; font-weight:600; }
.stButton>button:hover { transform:translateY(-1px);}
.viewer { background:#000; padding:10px; border-radius:16px; box-shadow:0 16px 32px rgba(0,0,0,0.3);}
.status-ready { background-color:#e8f5ee; color:#137333; padding:6px 12px; border-radius:12px; font-weight:600; }
.status-wait { background-color:#fff4e5; color:#8a5700; padding:6px 12px; border-radius:12px; font-weight:600;}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE INIT
# =====================================================
for k, v in {
    "image": None, "mask": None, "inference_requested": False,
    "inference_done": False, "patch_size_used": None,
    "step_used": None, "last_uploaded_name": None, "zoom":1.0
}.items():
    if k not in st.session_state: st.session_state[k] = v

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("⚙️ Controls")
uploaded = st.sidebar.file_uploader("Upload Dental X-ray", type=["png","jpg","jpeg"])

with st.sidebar.expander("Inference Settings", expanded=True):
    patch_size = st.selectbox("Patch Size", [256, 512, 1024], index=2)
    step = st.selectbox("Step Size", [64,128,192,256,320,384,448,512], index=1)
    if st.button("🚀 Run Inference"):
        st.session_state["inference_requested"] = True

with st.sidebar.expander("Visualization", expanded=True):
    class_choice = st.selectbox("Classes", ["Pulp","Enamel","Pulp + Enamel"], index=2)
    alpha = st.slider("Transparency", 0.1, 1.0, 0.5)
    show_overlay = st.checkbox("Overlay", True)
    show_mask = st.checkbox("Contours", True)
    st.slider("Zoom", 1.0, 3.0, st.session_state["zoom"], step=0.1, key="zoom_slider")

# =====================================================
# STATUS BAR
# =====================================================
if st.session_state["image"] is None:
    st.markdown('<span class="status-wait">Upload an image to begin</span>', unsafe_allow_html=True)
elif not st.session_state["inference_done"]:
    st.markdown('<span class="status-wait">Ready for inference</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-ready">Inference complete</span>', unsafe_allow_html=True)

# =====================================================
# MODEL (cached)
# =====================================================
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# =====================================================
# HANDLE UPLOAD
# =====================================================
if uploaded and st.session_state["last_uploaded_name"] != uploaded.name:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state["image"] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.session_state.update({"mask": None, "inference_done": False, "inference_requested": False, "last_uploaded_name": uploaded.name, "zoom":1.0})

# =====================================================
# RUN INFERENCE
# =====================================================
if st.session_state["inference_requested"] and st.session_state["image"] is not None:
    with st.spinner("Running inference..."):
        mask = predict_large_image(st.session_state["image"], model, patch_size, step)
    st.session_state.update({"mask": mask, "inference_done": True, "patch_size_used": patch_size, "step_used": step, "inference_requested": False})

# =====================================================
# DISPLAY IMAGE
# =====================================================
if st.session_state["image"] is not None:
    display_img = st.session_state["image"].copy()
    if st.session_state["inference_done"]:
        filtered_mask = filter_mask_by_class(st.session_state["mask"], class_choice)
        if show_overlay:
            display_img = create_overlay(display_img, filtered_mask, alpha=alpha)
        if show_mask:
            contours,_ = cv2.findContours((filtered_mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255,255,0), 1)


    zoom_factor = st.session_state.get("zoom_slider", 1.0)
    display_img = apply_zoom(display_img, zoom_factor)

  #  display_img = apply_zoom(display_img, st.session_state.get("zoom_slider", 1.0))
    st.image(display_img, use_column_width=True)
