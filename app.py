# import cv2
# import numpy as np
# import streamlit as st
# from infer import load_model, predict_large_image
# from utils import create_overlay, filter_mask_by_class, apply_zoom



# # =====================================================
# # CONFIG
# # =====================================================
# st.set_page_config(
#     page_title="Dental X-ray Segmentation",
#     layout="wide"
# )

# st.markdown(
#     "<h1 class='app-title'>🦷 Dental X-ray Segmentation</h1>",
#     unsafe_allow_html=True
# )
# st.caption("Patch-based semantic segmentation for dental radiographs")

# # =====================================================
# # CUSTOM CSS FOR AESTHETIC UI
# # =====================================================
# st.markdown("""
# <style>

# /* ===========================
#    GLOBAL
# =========================== */
# html, body, [class*="css"] {
#     font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
#     background-color: #f5f7fb;
# }

            
# /* ===========================
#    APP TITLE FIX
# =========================== */
# .app-title {
#     line-height: 1.2;
#     margin-top: 0.6rem;
#     margin-bottom: 0.4rem;
#     padding-top: 0.4rem;
# }

# /* ===========================
#    TITLE
# =========================== */
# h1 {
#     font-weight: 700;
#     letter-spacing: -0.6px;
#     margin-bottom: 0.25rem;
# }

# /* ===========================
#    SIDEBAR
# =========================== */
 
# section[data-testid="stSidebar"],
# section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
#     background-color: #0f172a !important;
#     color: #e5e7eb !important;
# }


# /* ===========================
#    BUTTONS
# =========================== */
# .stButton > button {
#     background: linear-gradient(135deg, #1f77b4, #155a8a);
#     color: white;
#     border-radius: 10px;
#     height: 46px;
#     font-weight: 600;
#     border: none;
#     box-shadow: 0 6px 16px rgba(31, 119, 180, 0.25);
#     transition: all 0.2s ease;
# }

# .stButton > button:hover {
#     transform: translateY(-1px);
#     box-shadow: 0 8px 20px rgba(31, 119, 180, 0.35);
# }

# /* ===========================
#    STATUS BADGES
# =========================== */
# .status {
#     padding: 6px 12px;
#     border-radius: 999px;
#     font-size: 0.85rem;
#     font-weight: 600;
#     display: inline-block;
#     margin-bottom: 0.8rem;
# }

# .status-ready {
#     background-color: #e8f5ee;
#     color: #137333;
#     border: 1px solid #b7e1cd;
# }

# .status-wait {
#     background-color: #fff4e5;
#     color: #8a5700;
#     border: 1px solid #ffd8a8;
# }

# /* ===========================
#    IMAGE VIEWER
# =========================== */
# .viewer {
#     background-color: #000;
#     padding: 14px;
#     border-radius: 16px;
#     box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
#     border: 1px solid rgba(255,255,255,0.08);
# }

# /* ===========================
#    MAIN AREA SPACING
# =========================== */
# .block-container {
#     padding-top: 2.2rem;
#     padding-bottom: 2rem;
# }


# /* ===========================
#    HIDE FOOTER
# =========================== */
# footer {
#     visibility: hidden;
# }

# </style>
# """, unsafe_allow_html=True)


# st.sidebar.markdown(
#     """
#     <div style="
#         text-align:center;
#         padding: 16px 0 8px 0;
#     ">
#         <img src="https://media.licdn.com/dms/image/v2/D4E0BAQH6y3Iz9pV40g/company-logo_200_200/company-logo_200_200/0/1727300800819?e=2147483647&v=beta&t=6CBv6JuZvgOtpr-4o6Q4OGIwhfaG3bNdOdCm3RfsM6M"
#              width="140"/>
#         <div style="
#             font-weight:700;
#             font-size:1rem;
#             margin-top:6px;
#         ">
#             SPAI LABS
#         </div>
#         <div style="
#             font-size:0.8rem;
#             color:#6b7280;
#         ">
#             Dental Imaging Platform
#         </div>
#     </div>
#     <hr>
#     """,
#     unsafe_allow_html=True
# )



# # =====================================================
# # SESSION STATE INIT
# # =====================================================
# state_defaults = {
#     "image": None,
#     "mask": None,
#     "inference_requested": False,
#     "inference_done": False,
#     "patch_size_used": None,
#     "step_used": None,
#     "last_uploaded_name": None,
#     "zoom": 1.0,
# }
# for k, v in state_defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # =====================================================
# # STATUS BAR
# # =====================================================
# if st.session_state["image"] is None:
#     st.markdown('<span class="status status-wait">Upload an image to begin</span>',
#                 unsafe_allow_html=True)
# elif not st.session_state["inference_done"]:
#     st.markdown('<span class="status status-wait">Ready for inference</span>',
#                 unsafe_allow_html=True)
# else:
#     st.markdown('<span class="status status-ready">Inference complete</span>',
#                 unsafe_allow_html=True)

# # =====================================================
# # SIDEBAR
# # =====================================================
# # BUTTON CALLBACK
# def request_inference():
#     st.session_state["inference_requested"] = True

# st.sidebar.markdown("### 🖼️ Input")
# uploaded = st.sidebar.file_uploader(
#     "Upload Dental X-ray",
#     type=["png", "jpg", "jpeg"],
#     label_visibility="collapsed"
# )

# st.sidebar.markdown("### ⚙️ Inference")
# patch_size = st.sidebar.selectbox("Patch Size", [256, 512, 1024], index=2)
# step = st.sidebar.selectbox("Step Size", [64, 128, 192, 256, 320, 384, 448, 512], index=1)

# st.sidebar.markdown("### 🎨 Visualization")
# class_choice = st.sidebar.selectbox(
#     "Classes",
#     ["Pulp", "Enamel", "Pulp + Enamel"],
#     index=2
# )
# alpha = st.sidebar.slider("Transparency", 0.1, 1.0, 0.5)
# show_overlay = st.sidebar.checkbox("Overlay", True)
# show_mask = st.sidebar.checkbox("Contours", True)


# st.sidebar.markdown("---")
# st.sidebar.button("🚀 Run Inference", on_click=request_inference)

# # =====================================================
# # MODEL (cached)
# # =====================================================
# @st.cache_resource
# def get_model():
#     return load_model()

# model = get_model()

# # =====================================================
# # HANDLE UPLOAD (NEW IMAGE ONLY)
# # =====================================================
# if uploaded:
#     if st.session_state["last_uploaded_name"] != uploaded.name:
#         file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
#         image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#         image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#         st.session_state["image"] = image_rgb
#         st.session_state["mask"] = None
#         st.session_state["inference_done"] = False
#         st.session_state["inference_requested"] = False
#         st.session_state["last_uploaded_name"] = uploaded.name
#         st.session_state["zoom"] = 1.0  # reset zoom

# # =====================================================
# # RUN INFERENCE (STRICTLY CONTROLLED)
# # =====================================================
# if (
#     st.session_state["inference_requested"]
#     and st.session_state["image"] is not None
#     and (
#         not st.session_state["inference_done"]
#         or st.session_state["patch_size_used"] != patch_size
#         or st.session_state["step_used"] != step
#     )
# ):
#     with st.spinner("Running inference (only once)..."):
#         mask = predict_large_image(
#             image_rgb=st.session_state["image"],
#             model=model,
#             patch_size=patch_size,
#             step=step,
#         )

#     st.session_state["mask"] = mask
#     st.session_state["inference_done"] = True
#     st.session_state["patch_size_used"] = patch_size
#     st.session_state["step_used"] = step
#     st.session_state["inference_requested"] = False

# # =====================================================
# # DISPLAY (IMAGE + ZOOM CONTROLS)
# # =====================================================
# if st.session_state["image"] is not None:

#     final_view = st.session_state["image"].copy()

#     if st.session_state["inference_done"]:
#         filtered_mask = filter_mask_by_class(
#             st.session_state["mask"],
#             class_choice
#         )

#         if show_overlay:
#             final_view = create_overlay(
#                 final_view,
#                 filtered_mask,
#                 alpha=alpha
#             )

#         if show_mask:
#             contours, _ = cv2.findContours(
#                 (filtered_mask > 0).astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )
#             cv2.drawContours(final_view, contours, -1, (255, 255, 0), 1)

#     # -------- ZOOM APPLY --------
    

#     col1, col2, col3 = st.columns([1, 4, 1])

#     with col2:
#         # ---- ZOOM BUTTONS (ADD HERE) ----
#         zc1, zc2, zc3 = st.columns([1, 2, 1])

#         with zc1:
#             if st.button("➖", key="zoom_out"):
#                 st.session_state["zoom"] = max(1.0, st.session_state["zoom"] - 0.2)

#         with zc3:
#             if st.button("➕", key="zoom_in"):
#                 st.session_state["zoom"] = min(3.0, st.session_state["zoom"] + 0.2)


#         zoomed_view = apply_zoom(final_view, st.session_state["zoom"])
#         # ---- IMAGE VIEWER ----
#         st.markdown('<div class="viewer">', unsafe_allow_html=True)
#         st.image(zoomed_view)   # ❗ NO use_column_width
#         st.markdown('</div>', unsafe_allow_html=True)

# import cv2
# import numpy as np
# import streamlit as st
# from infer import load_model, predict_large_image
# from utils import create_overlay, filter_mask_by_class, apply_zoom

# st.set_page_config(page_title="Dental X-ray Segmentation", layout="wide")

# st.markdown("""
#     <style>
#     .main-header { text-align: center; color: #1f2937; font-size: 3rem; font-weight: 800; margin: 2rem 0; }
#     .subtitle { text-align: center; color: #6b7280; font-size: 1.1rem; margin-bottom: 3rem; }
#     .upload-area { background: linear-gradient(135deg, #f8fafc, #e2e8f0); border-radius: 16px; padding: 2rem; text-align: center; border: 2px dashed #cbd5e1; }
#     .control-panel { background: #f1f5f9; border-radius: 12px; padding: 1.5rem; margin: 1rem 0; }
#     .viewer-container { background: #000; border-radius: 16px; padding: 1rem; box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }
#     .status-badge { padding: 0.5rem 1rem; border-radius: 50px; font-weight: 600; font-size: 0.9rem; display: inline-block; }
#     .status-ready { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
#     .status-processing { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
#     .sidebar .css-1d391kg { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
#     .stButton > button { background: linear-gradient(135deg, #3b82f6, #1d4ed8); border-radius: 12px; height: 48px; font-weight: 700; box-shadow: 0 10px 15px rgba(59,130,246,0.3); }
#     .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 20px rgba(59,130,246,0.4); }
#     .stSlider > div > div > div > div { background: #3b82f6; }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown('<h1 class="main-header">🦷 Dental X-ray Segmentation</h1>', unsafe_allow_html=True)
# st.markdown('<p class="subtitle">Advanced patch-based semantic segmentation for precise dental analysis</p>', unsafe_allow_html=True)

# # Session state init
# state_defaults = {
#     "image": None, "mask": None, "inference_done": False, "inference_requested": False,
#     "patch_size_used": None, "step_used": None, "last_uploaded_name": None, "zoom": 1.0
# }
# for k, v in state_defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # Status
# status = "Upload an image to start" if st.session_state.image is None else ("Ready for inference" if not st.session_state.inference_done else "Inference complete ✓")
# st.markdown(f'<span class="status-badge status-{"processing" if "Ready" in status else "ready"}">{status}</span>', unsafe_allow_html=True)

# # Cached model
# @st.cache_resource
# def get_model():
#     return load_model()
# model = get_model()

# # Sidebar
# with st.sidebar:
#     st.markdown("""
#         <div style="text-align:center; padding:1rem 0;">
#             <img src="https://media.licdn.com/dms/image/v2/D4E0BAQH6y3Iz9pV40g/company-logo_200_200/company-logo_200_200/0/1727300800819?e=2147483647&v=beta&t=6CBv6JuZvgOtpr-4o6Q4OGIwhfaG3bNdOdCm3RfsM6M" width="120"/>
#             <div style="font-weight:700; font-size:1.1rem; margin-top:0.5rem;">SPAI LABS</div>
#             <div style="font-size:0.85rem; color:#9ca3af;">Dental Imaging Platform</div>
#         </div>
#         <hr>
#     """, unsafe_allow_html=True)
    
#     st.markdown("### 🖼️ Input")
#     uploaded = st.file_uploader("Upload Dental X-ray", type=['png','jpg','jpeg'])
    
#     st.markdown("### ⚙️ Inference")
#     col1, col2 = st.columns(2)
#     with col1: patch_size = st.selectbox("Patch Size", [256,512,1024], index=2)
#     with col2: step = st.selectbox("Step Size", [64,128,192,256,320,384,448,512], index=1)
    
#     st.markdown("### 🎨 Visualization")
#     class_choice = st.selectbox("Classes", ["Pulp","Enamel","Pulp + Enamel"], index=2)
#     alpha = st.slider("Transparency", 0.1, 1.0, 0.5)
#     show_overlay = st.checkbox("Overlay", True)
#     show_contours = st.checkbox("Contours", True)
    
#     st.markdown("---")
#     def request_inference():
#         st.session_state["inference_requested"] = True
#     st.button("🚀 Run Inference", on_click=request_inference, use_container_width=True)

# # Handle upload
# if uploaded and st.session_state["last_uploaded_name"] != uploaded.name:
#     file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
#     image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     st.session_state["image"] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     st.session_state["mask"] = None
#     st.session_state["inference_done"] = False
#     st.session_state["inference_requested"] = False
#     st.session_state["last_uploaded_name"] = uploaded.name
#     st.session_state["zoom"] = 1.0

# # Run inference
# if (
#     st.session_state["inference_requested"]
#     and st.session_state["image"] is not None
#     and (
#         not st.session_state["inference_done"]
#         or st.session_state["patch_size_used"] != patch_size
#         or st.session_state["step_used"] != step
#     )
# ):
#     with st.spinner("Running inference..."):
#         mask = predict_large_image(st.session_state["image"], model, patch_size, step)
#     st.session_state["mask"] = mask
#     st.session_state["inference_done"] = True
#     st.session_state["patch_size_used"] = patch_size
#     st.session_state["step_used"] = step
#     st.session_state["inference_requested"] = False

# # Display
# if st.session_state["image"] is not None:
#     final_view = st.session_state["image"].copy()
#     if st.session_state["inference_done"] and st.session_state["mask"] is not None:
#         filtered_mask = filter_mask_by_class(st.session_state["mask"], class_choice)
#         if show_overlay:
#             final_view = create_overlay(final_view, filtered_mask, alpha=alpha)
#         if show_contours:
#             contours, _ = cv2.findContours((filtered_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(final_view, contours, -1, (255, 255, 0), 2)
    
#     # Zoom
#     colz1, colz2, colz3 = st.columns([1,3,1])
#     with colz1:
#         if st.button("➖", key="zoom_out"): st.session_state["zoom"] = max(0.5, st.session_state["zoom"] - 0.25)
#     with colz2: st.write(f"Zoom: {st.session_state['zoom']:.1f}x")
#     with colz3:
#         if st.button("➕", key="zoom_in"): st.session_state["zoom"] = min(4.0, st.session_state["zoom"] + 0.25)
    
#     zoomed_view = apply_zoom(final_view, st.session_state["zoom"])
#     st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
#     st.image(zoomed_view, use_column_width=True)
#     st.markdown('</div>', unsafe_allow_html=True)




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
