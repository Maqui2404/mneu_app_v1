# pip install streamlit tensorflow pillow opencv-python-headless numpy
# streamlit run app_pneumonia_streamlit.py
# Create a clean minimal Streamlit app for the prototype with Grad-CAM / Grad-CAM++
# from pathlib import Path

# app_pneumonia_streamlit.py
# Minimal, clean Streamlit prototype for chest X-ray pneumonia detection
# Features:
# - Load Keras .h5 or .keras model (DenseNet121 head as in training)
# - Upload JPG/PNG (grayscale OK), preprocess (224, RGB, /255.0)
# - Predict probability, decision with adjustable threshold
# - Grad-CAM or Grad-CAM++ overlay with opacity control
# - Auto-detect target conv layer (prefers 'conv5_block16_concat')
# - Download overlay PNG
#
# Run:
#   streamlit run app_pneumonia_streamlit.py
#
# Requirements:
#   pip install streamlit tensorflow pillow opencv-python-headless numpy

import io
import os
import cv2
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Pneumonia XR Prototype", page_icon="🫁", layout="wide")

# -----------------------------
# Sidebar: configuration
# -----------------------------
st.sidebar.header("⚙️ Configuración")
model_path = st.sidebar.text_input("Ruta del modelo (.h5 o .keras)", value="best_model.h5")
img_size = st.sidebar.selectbox("Tamaño de entrada", options=[224, 256, 320], index=0)
method = st.sidebar.selectbox("Método de mapa", options=["Grad-CAM", "Grad-CAM++"], index=0)
alpha = st.sidebar.slider("Opacidad del overlay", 0.10, 0.80, 0.35, 0.05)
threshold = st.sidebar.slider("Umbral de decisión (Prob. PNEUMONIA)", 0.10, 0.90, 0.50, 0.01)
st.sidebar.caption("Nota: Prototipo de investigación. No reemplaza juicio clínico.")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str, img_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el archivo de modelo: {model_path}")
    # Try to load full model first; if fails, build architecture and load by name
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        return m
    except Exception:
        # Rebuild the architecture used in training, then load by name
        base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
                                                 input_shape=(img_size, img_size, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        m = tf.keras.Model(base.input, out)
        m(np.zeros((1, img_size, img_size, 3), dtype=np.float32))  # build
        m.load_weights(model_path, by_name=True, skip_mismatch=True)
        return m

def detect_target_layer(m):
    pref = "conv5_block16_concat"
    names = [l.name for l in m.layers]
    if pref in names:
        return pref
    # fallback: last 4D output layer
    for layer in reversed(m.layers):
        try:
            out = layer.output
            if len(out.shape) == 4:
                return layer.name
        except Exception:
            continue
    # last resort: last Conv2D
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró una capa válida para CAM.")

def read_and_preprocess(img_file, size: int):
    # read with PIL for robustness
    pil = Image.open(img_file)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((size, size))
    rgb = np.array(pil).astype(np.float32) / 255.0  # training used /255.0
    return pil, rgb

def _forward_for_cam(m, img_norm, last_conv):
    x = np.expand_dims(img_norm, 0)
    gm = tf.keras.models.Model([m.inputs], [m.get_layer(last_conv).output, m.output])
    conv_out, preds = gm(x)
    p = tf.clip_by_value(preds[:, 0], 1e-6, 1-1e-6)  # probability of positive class
    return conv_out, p

def grad_cam(m, img_norm, last_conv):
    with tf.GradientTape() as tape:
        conv_out, p = _forward_for_cam(m, img_norm, last_conv)
        logit = tf.math.log(p/(1.0 - p))
    grads = tape.gradient(logit, conv_out)
    weights = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
    fmap = conv_out[0].numpy()
    cam = np.dot(fmap, weights)
    cam = np.maximum(cam, 0); cam = cam / (cam.max() + 1e-8)
    return cam, p.numpy().item()

def grad_cam_pp(m, img_norm, last_conv):
    x = np.expand_dims(img_norm, 0)
    gm = tf.keras.models.Model([m.inputs], [m.get_layer(last_conv).output, m.output])
    with tf.GradientTape(persistent=True) as t1:
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape() as t3:
                conv_out, preds = gm(x)
                p = tf.clip_by_value(preds[:, 0], 1e-6, 1-1e-6)
                logit = tf.math.log(p/(1.0 - p))
            g1 = t3.gradient(logit, conv_out)
        g2 = t2.gradient(g1, conv_out)
    g3 = t1.gradient(g2, conv_out)

    conv = conv_out[0]; grad = g1[0]; grad2 = g2[0]; grad3 = g3[0]
    eps = 1e-7
    denom = 2.0*grad2 + conv*grad3
    denom = tf.where(tf.abs(denom) < eps, tf.ones_like(denom)*eps, denom)
    alphas = grad2 / denom
    weights = tf.reduce_sum(alphas * tf.nn.relu(grad), axis=(0,1)).numpy()
    cam = np.dot(conv.numpy(), weights)
    cam = np.maximum(cam, 0); cam = cam / (cam.max() + 1e-8)
    return cam, p.numpy().item()

def make_overlay(cam, base_rgb, alpha=0.35):
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (base_rgb.shape[1], base_rgb.shape[0])), cv2.COLORMAP_JET)
    overlay = (alpha*heat + (1.0 - alpha)*(base_rgb*255)).astype(np.uint8)
    return overlay

# -----------------------------
# Header
# -----------------------------
st.title("🫁 Pneumonia XR Prototype")
st.caption("Detección asistida con Grad-CAM/Grad-CAM++. Prototipo de investigación — no uso clínico.")

# -----------------------------
# Load model
# -----------------------------
model = None
try:
    with st.spinner("Cargando modelo..."):
        model = load_model(model_path, img_size)
    st.success("Modelo cargado.")
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")

# -----------------------------
# Upload & Predict
# -----------------------------
left, right = st.columns([1,1])

with left:
    uploaded = st.file_uploader("Sube una radiografía (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded is not None and model is not None:
    pil, img_norm = read_and_preprocess(uploaded, img_size)
    target_layer = detect_target_layer(model)

    # show original
    with left:
        st.subheader("Imagen")
        st.image(pil, use_column_width=True)

    # CAM
    try:
        if method == "Grad-CAM++":
            cam, prob = grad_cam_pp(model, img_norm, target_layer)
        else:
            cam, prob = grad_cam(model, img_norm, target_layer)
    except Exception as e:
        st.error(f"Error generando CAM: {e}")
        st.stop()

    overlay = make_overlay(cam, np.array(pil).astype(np.float32)/255.0, alpha)

    # decision
    decision = "PNEUMONIA" if prob >= threshold else "NORMAL"

    with right:
        st.subheader("Resultado")
        st.metric("Prob(PNEUMONIA)", f"{prob:.3f}")
        st.write(f"**Decisión (umbral {threshold:.2f})**: **{decision}**")
        st.subheader(method)
        st.image(overlay, use_column_width=True, caption="Regiones más influyentes")

        # Downloads
        st.download_button(
            "⬇️ Descargar overlay PNG",
            data=cv2.imencode(".png", overlay[:, :, ::-1])[1].tobytes(),
            file_name="gradcam_overlay.png",
            mime="image/png",
        )
        # JSON dump
        result = {"prob_pneumonia": float(prob), "threshold": float(threshold), "decision": decision,
                  "method": method, "target_layer": target_layer}
        st.download_button(
            "⬇️ Descargar resultado JSON",
            data=json.dumps(result, indent=2).encode("utf-8"),
            file_name="resultado.json",
            mime="application/json",
        )
else:
    st.info("Carga un modelo válido y luego sube una imagen para analizar.")

# Footer
st.markdown("---")
st.caption("© Prototype • DenseNet121 • Preproc: gris→RGB, 224, /255.0 • CAM layer auto")
