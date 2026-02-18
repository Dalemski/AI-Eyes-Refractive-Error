from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# -----------------------------
# App config (must be first Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="AI Retinal Image Analysis",
    layout="centered",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Grad-CAM (binary ResNet18)
# -----------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activations)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        target_class: 1 -> Myopia, 0 -> Normal (binary logits).
        Returns heatmap in [0,1] at feature-map resolution (e.g., 7x7).
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)  # expected [1,1]
        if logits.ndim != 2 or logits.shape[1] != 1:
            raise ValueError(f"Expected model output shape [B,1], got {tuple(logits.shape)}")

        score = logits[0, 0] if target_class == 1 else -logits[0, 0]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Check target_layer.")

        acts = self.activations  # [1, C, H, W]
        grads = self.gradients   # [1, C, H, W]

        weights = grads.mean(dim=(2, 3), keepdim=True)      # [1, C, 1, 1]
        cam = (weights * acts).sum(dim=1).squeeze(0)        # [H, W]
        cam = cam.detach().cpu().numpy()

        cam = np.maximum(cam, 0)
        denom = cam.max() - cam.min()
        if denom < 1e-8:
            return np.zeros_like(cam, dtype=np.float32)
        cam = (cam - cam.min()) / denom
        return cam.astype(np.float32)

    def close(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()


# -----------------------------
# Demo images (random picker)
# -----------------------------
DEMO_DIR = Path(__file__).parent / "demo_images"
DEMO_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_demo_images() -> list[Path]:
    if not DEMO_DIR.exists() or not DEMO_DIR.is_dir():
        return []
    files = [p for p in DEMO_DIR.iterdir() if p.is_file() and p.suffix.lower() in DEMO_EXTS]
    files.sort()
    return files

def pick_random_demo_image() -> Path | None:
    files = list_demo_images()
    if not files:
        return None
    return random.choice(files)

def demo_dir_label() -> str:
    return "demo_images/"


# -----------------------------
# Retina-only gate (heuristic)
# -----------------------------
def _to_bgr_uint8(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def retina_gate_score(pil_img: Image.Image) -> tuple[float, dict]:
    """
    Heuristic retinal/fundus detector.
    Returns (score in [0,1], details dict).

    Signals:
    - Fundus: brighter-ish center, darker corners, reddish bias, fewer text-like edges.
    - Documents/screens: higher edge density, different brightness geometry.
    """
    bgr = _to_bgr_uint8(pil_img)
    h, w = bgr.shape[:2]

    # Downscale for speed
    scale = 512.0 / max(h, w) if max(h, w) > 512 else 1.0
    if scale != 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hh, ww = gray.shape[:2]

    yy, xx = np.mgrid[0:hh, 0:ww]
    cx, cy = ww / 2.0, hh / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_norm = r / (r.max() + 1e-8)

    center_mask = r_norm <= 0.45
    corner_mask = r_norm >= 0.75

    center_mean = float(gray[center_mask].mean()) if center_mask.any() else 0.0
    corner_mean = float(gray[corner_mask].mean()) if corner_mask.any() else 0.0
    brightness_contrast = (center_mean - corner_mean) / 255.0  # ~[0..1]

    edges = cv2.Canny(gray, 60, 140)
    edge_density = float((edges > 0).mean())  # 0..1
    edge_score = 1.0 - np.clip(edge_density / 0.12, 0.0, 1.0)

    b, g, rch = cv2.split(bgr)
    red_bias = float((rch.mean() - g.mean()) / 255.0)
    red_score = np.clip((red_bias + 0.05) / 0.25, 0.0, 1.0)

    score = (
        0.45 * np.clip((brightness_contrast + 0.02) / 0.25, 0.0, 1.0)
        + 0.35 * edge_score
        + 0.20 * red_score
    )
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "center_mean": center_mean,
        "corner_mean": corner_mean,
        "brightness_contrast": float(brightness_contrast),
        "edge_density": float(edge_density),
        "red_bias": float(red_bias),
        "score": score,
        "resized_shape": (hh, ww),
    }
    return score, details

def is_likely_retinal(pil_img: Image.Image, threshold: float) -> tuple[bool, dict]:
    score, details = retina_gate_score(pil_img)
    return (score >= threshold), details


# -----------------------------
# Heatmap / explainability helpers
# -----------------------------
def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def _heatmap_to_colormap_uint8(heatmap_01: np.ndarray) -> np.ndarray:
    hm = np.clip(heatmap_01, 0.0, 1.0)
    hm_u8 = (hm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)  # BGR uint8

def overlay_heatmap_on_rgb(img_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    heat_color_bgr = _heatmap_to_colormap_uint8(heatmap_01)
    heat_color_rgb = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, heat_color_rgb, alpha, 0)

def find_topk_hotspots_bboxes(
    heatmap_01: np.ndarray,
    k: int = 3,
    thr: float = 0.6,
    min_area: int = 80,
) -> list[tuple[int, int, int, int, float]]:
    """
    Peak-based hotspots: find up to k local maxima, then create fixed-size boxes around peaks.
    Returns: (x, y, w, h, score) in image coordinates.
    """
    hm = np.asarray(heatmap_01, dtype=np.float32)
    hm = np.clip(hm, 0.0, 1.0)
    H, W = hm.shape[:2]

    if H == 0 or W == 0 or float(hm.max()) < 1e-6:
        return []

    thr = float(np.clip(thr, 0.0, 1.0))

    # Detect local maxima via dilation
    ksize = max(7, (int(round(min(H, W) * 0.05)) | 1))  # odd kernel, ~5% of min side
    kernel = np.ones((ksize, ksize), np.uint8)
    dil = cv2.dilate(hm, kernel)

    peaks = (hm >= thr) & (hm >= (dil - 1e-6))
    ys, xs = np.where(peaks)
    if ys.size == 0:
        return []

    scores = hm[ys, xs]
    order = np.argsort(scores)[::-1]

    box_w = max(12, int(round(W * 0.18)))
    box_h = max(12, int(round(H * 0.18)))

    selected: list[tuple[int, int, int, int, float]] = []
    taken = np.zeros((H, W), dtype=np.uint8)

    for idx in order:
        if len(selected) >= int(k):
            break
        x = int(xs[idx])
        y = int(ys[idx])
        if taken[y, x]:
            continue

        x0 = max(0, x - box_w // 2)
        y0 = max(0, y - box_h // 2)
        x1 = min(W, x0 + box_w)
        y1 = min(H, y0 + box_h)

        taken[y0:y1, x0:x1] = 1

        region = hm[y0:y1, x0:x1]
        score = float(region.max()) if region.size else float(hm[y, x])
        area = int((y1 - y0) * (x1 - x0))
        if area < int(min_area):
            continue

        selected.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), score))

    return selected

def draw_bboxes_on_rgb(
    img_rgb: np.ndarray,
    bboxes: list[tuple[int, int, int, int, float]],
    color_rgb: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = img_rgb.copy()
    for i, (x, y, w, h, score) in enumerate(bboxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), color_rgb, 2)
        cv2.putText(
            out,
            f"{i}: {score:.2f}",
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color_rgb,
            2,
            lineType=cv2.LINE_AA,
        )
    return out


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model(model_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)  # binary logit

    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    model.eval()
    return model


MODEL_PATH = "myopia_model.pth"
model = load_model(MODEL_PATH)
TARGET_LAYER = model.layer4[-1].conv2


# -----------------------------
# Preprocess (must match training)
# -----------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Change mean/std if your training used ImageNet normalization.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# -----------------------------
# UI
# -----------------------------
st.title("AI Retinal Image Analysis")
st.write(
    "Upload a retinal (fundus) image and the model will predict **Normal** vs **Myopia**. "
    "Enable Grad-CAM to visualize the most influential regions."
)

with st.sidebar:
    if st.button("About this app"):
        if "show_about" not in st.session_state:
            st.session_state["show_about"] = True
        else:
            st.session_state["show_about"] = not st.session_state["show_about"]

    if st.session_state.get("show_about", False):
        st.info(
            "This application leverages deep learning to analyze retinal (fundus) images. "
            "It predicts whether the image indicates **Normal Vision** or **Refractive Error**. "
            "The app also provides Grad-CAM visualizations to highlight the most influential regions "
            "in the image contributing to the prediction. "
        )
    st.header("Controls")
    
    show_gradcam = st.toggle("Show Grad-CAM", value=True, help="Enable Grad-CAM visualization to highlight influential regions in the image.")
    overlay_strength = st.slider("Heatmap overlay strength", 0.0, 1.0, 0.45, 0.05, help="Adjust the transparency of the heatmap overlay on the image.")
    confidence_warn_threshold = st.slider("Low-confidence warning threshold", 0.50, 0.90, 0.55, 0.01, help="Set the confidence threshold below which a warning is displayed.")

    st.divider()
    st.subheader("Retina-only filter")
    enable_retina_gate = st.toggle("Reject non-retinal images", value=True, help="Enable a filter to reject images that do not resemble retinal photos.")
    retina_gate_threshold = st.slider("Gate threshold", 0.10, 0.90, 0.45, 0.01, help="Set the threshold for the retina-only filter. Higher values make the filter stricter.")

    st.divider()
    st.subheader("Hotspot boxes")
    hotspot_k = st.slider("Top hotspots (k)", 1, 5, 3, 1, help="Select the number of top hotspots to highlight in the image.")
    hotspot_thr = st.slider("Hotspot threshold", 0.10, 0.95, 0.60, 0.01, help="Set the threshold for hotspot activation. Higher values show fewer hotspots.")
    hotspot_min_area = st.slider("Min hotspot area", 10, 2000, 80, 10, help="Set the minimum area for a hotspot to be considered valid.")
    show_debug = st.checkbox("Show debug info", value=False, help="Enable debug information for developers.")



# -----------------------------
# Main flow (upload or persistent demo)
# -----------------------------
st.subheader("Try a demo image")
col_a, col_b = st.columns([1, 2])

with col_a:
    if st.button("Random demo image"):
        picked_image = pick_random_demo_image()
        if picked_image is None:
            st.warning(f"No demo images found. Put images in `{demo_dir_label()}`")
            st.session_state.pop("demo_path", None)
        else:
            st.session_state["demo_path"] = str(picked_image )

with col_b:
    demo_dir_label()

uploaded_file = st.file_uploader("Upload a Retinal Image", type=["jpg", "png", "jpeg"])

demo_path = Path(st.session_state["demo_path"]) if st.session_state.get("demo_path") else None

# If user uploads, prefer upload and clear demo selection to avoid confusion
if uploaded_file is not None:
    st.session_state.pop("demo_path", None)
    demo_path = None

if uploaded_file is None and demo_path is None:
    st.info("Upload an image or click **Random demo image** to begin.")
else:
    try:
        st.markdown(
                """
                <div style="color: #FF6347; font-size: 0.9em; margin-top: 20px;">
                    **Disclaimer:** This application uses AI to analyze retinal images and provide predictions. 
                    The results are not a substitute for professional medical advice, diagnosis, or treatment. 
                    Always consult a qualified healthcare provider for medical concerns. Do not rely solely on this tool for making decisions about prescriptions or treatments.
                </div>
                """,
                unsafe_allow_html=True,
            )
        # Load image
        if demo_path is not None:
            image = Image.open(demo_path).convert("RGB")
            if show_debug:
                st.write(f"Demo image: {demo_path.name}")
        else:
            if show_debug:
                st.write(f"File name: {uploaded_file.name}")
                st.write(f"File type: {uploaded_file.type}")
                st.write(f"File size: {uploaded_file.size} bytes")
            image = Image.open(uploaded_file).convert("RGB")

        img_rgb = pil_to_rgb_np(image)
        st.image(img_rgb, caption="Selected Image", width=500)

        # Retina-only gate
        if enable_retina_gate:
            ok, gate_details = is_likely_retinal(image, threshold=float(retina_gate_threshold))
            if show_debug:
                st.write("Retina gate details:", gate_details)
            if not ok:
                st.error(
                    "This image does not look like a retinal/fundus photo. "
                    "Prediction was blocked. Please upload a clear retinal image."
                )
                st.stop()

        input_tensor = preprocess(image).unsqueeze(0)  # [1,3,224,224]
        if show_debug:
            st.write(f"Input tensor shape: {input_tensor.shape}")
            st.write(f"Model input shape: {input_tensor.shape}")

        # Prediction
        with torch.no_grad():
            logits = model(input_tensor)  # [1,1]
            prob_myopia = torch.sigmoid(logits).item()

        if prob_myopia > 0.5:
            label = "Myopia"
            confidence = prob_myopia
            target_class = 1
        else:
            label = "Normal"
            confidence = 1.0 - prob_myopia
            target_class = 0

        st.subheader("Prediction")
        st.write(f"**{label}** (Confidence: **{confidence * 100:.2f}%**)")

        if confidence < float(confidence_warn_threshold):
            st.warning("Low confidence prediction. Try a clearer/centered retinal image.")

        # Grad-CAM (side-by-side + boxes)
        if show_gradcam:
            st.subheader("Explainability (Grad-CAM)")
            try:
                grad_cam = GradCAM(model, TARGET_LAYER)
                heatmap_small = grad_cam.generate_cam(input_tensor, target_class=target_class)
                grad_cam.close()

                h, w = img_rgb.shape[:2]
                heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_CUBIC)
                heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

                heatmap_color_rgb = cv2.cvtColor(_heatmap_to_colormap_uint8(heatmap), cv2.COLOR_BGR2RGB)
                overlay_rgb = overlay_heatmap_on_rgb(img_rgb, heatmap, alpha=float(overlay_strength))

                bboxes = find_topk_hotspots_bboxes(
                    heatmap,
                    k=int(hotspot_k),
                    thr=float(hotspot_thr),
                    min_area=int(hotspot_min_area),
                )
                boxed_overlay = draw_bboxes_on_rgb(overlay_rgb, bboxes, color_rgb=(0, 255, 0))

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_rgb, caption="Original", width=240)
                with c2:
                    st.image(heatmap_color_rgb, caption="Heatmap", width=240)
                with c3:
                    st.image(overlay_rgb, caption="Overlay", width=240)

                st.image(boxed_overlay, caption="Top activated regions (boxed)", width=500)

                if show_debug:
                    st.write(f"Grad-CAM heatmap shape (small): {heatmap_small.shape}")
                    st.write(f"Hotspot boxes: {bboxes}")

            except Exception as e:
                st.error("Grad-CAM failed to generate.")
                if show_debug:
                    st.write(f"Error details: {e}")

    except UnidentifiedImageError:
        st.error("Invalid file. Please upload a valid image (jpg/png).")
    except Exception as e:
        st.error("Something went wrong while processing the image.")
        if show_debug:
            st.write(f"Error details: {e}")

            # Disclaimer
            

# Footer (optional)
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    .custom-footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    <div class="custom-footer">
        Created by <a href="https://www.linkedin.com/in/ivan-dalemski-449a27295/" target="_blank">Ivan Dalemski</a>
    </div>
    """,
    unsafe_allow_html=True,
)