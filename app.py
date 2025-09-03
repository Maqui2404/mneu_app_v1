import io
import os
import cv2
import json
import time
import shutil
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple

st.set_page_config(page_title="Pneumonia XR Prototype+", page_icon="🫁", layout="wide")

# -----------------------------
# Sidebar: configuration
# -----------------------------
st.sidebar.header("⚙️ Configuración")
model_path = st.sidebar.text_input("Ruta del modelo (.h5 o .keras)", value="best_model.h5")
img_size = st.sidebar.selectbox("Tamaño de entrada", options=[224, 256, 320], index=0)

# Decision Modes
mode = st.sidebar.selectbox("🧭 Modo de decisión", ["Tamizaje (↑sens)", "Estándar", "Confirmación (↑prec)"], index=1)
default_thr = 0.30 if "Tamizaje" in mode else (0.70 if "Confirmación" in mode else 0.50)
threshold = st.sidebar.slider("Umbral de decisión", 0.10, 0.90, float(default_thr), 0.01)

# CAM config
method_default = st.sidebar.selectbox("Método de mapa por defecto", ["Grad-CAM", "Grad-CAM++"], index=0)
alpha = st.sidebar.slider("Opacidad del overlay", 0.10, 0.80, 0.35, 0.05)

# Image Quality options
st.sidebar.subheader("🧪 Calidad de imagen")
enable_crop = st.sidebar.checkbox("Recortar bordes negros", value=True)
dark_cut = st.sidebar.slider("Umbral de negro (%)", 0, 100, 50, 1)
bright_cut = st.sidebar.slider("Umbral de blanco (%)", 0, 100, 50, 1)

st.sidebar.caption("Prototipo de investigación • No uso clínico.")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str, img_size: int):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe el archivo de modelo: {model_path}")
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        return m
    except Exception:
        base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet',
                                                 input_shape=(img_size, img_size, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        m = tf.keras.Model(base.input, out)
        _ = m.predict(np.zeros((1, img_size, img_size, 3), dtype=np.float32), verbose=0)
        m.load_weights(model_path, by_name=True, skip_mismatch=True)
        return m

def detect_target_layer(m):
    pref = "conv5_block16_concat"
    names = [l.name for l in m.layers]
    if pref in names:
        return pref
    for layer in reversed(m.layers):
        try:
            out = layer.output
            if len(out.shape) == 4:
                return layer.name
        except Exception:
            continue
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No se encontró una capa válida para CAM.")

def crop_black_borders(pil_img: Image.Image, pad=3) -> Image.Image:
    # Simple threshold-based crop for black borders
    gray = np.array(pil_img.convert("L"))
    mask = gray > 5  # non-black
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    y0 = max(y0 - pad, 0); x0 = max(x0 - pad, 0)
    y1 = min(y1 + pad, gray.shape[0]); x1 = min(x1 + pad, gray.shape[1])
    return pil_img.crop((x0, y0, x1, y1))

def read_and_preprocess(img_file, size: int, do_crop: bool = True):
    pil = Image.open(img_file).convert("RGB")
    if do_crop:
        pil = crop_black_borders(pil)
    pil = pil.resize((size, size))
    rgb = np.array(pil).astype(np.float32) / 255.0
    return pil, rgb

def _forward_for_cam(m, img_norm, last_conv):
    x = np.expand_dims(img_norm, 0)
    gm = tf.keras.models.Model([m.inputs], [m.get_layer(last_conv).output, m.output])
    conv_out, preds = gm(x)
    p = tf.clip_by_value(preds[:, 0], 1e-6, 1-1e-6)
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
    hmap = (cam * 255).astype(np.uint8)
    hmap = cv2.applyColorMap(cv2.resize(hmap, (base_rgb.shape[1], base_rgb.shape[0])), cv2.COLORMAP_JET)
    overlay = (alpha*hmap + (1.0 - alpha)*(base_rgb*255)).astype(np.uint8)
    return overlay

def image_quality_checks(pil_img: Image.Image, dark_cut: int, bright_cut: int):
    gray = np.array(pil_img.convert("L"))
    p_dark = float((gray < 20).mean())
    p_bright = float((gray > 235).mean())
    issues = []
    if p_dark*100 > dark_cut:
        issues.append(f"Subexposición (zona muy oscura: {p_dark*100:.1f}%)")
    if p_bright*100 > bright_cut:
        issues.append(f"Sobreexposición (zona muy brillante: {p_bright*100:.1f}%)")
    return issues, p_dark, p_bright

# Batch helpers
def extract_zip_to_temp(zf: bytes) -> str:
    tmp_dir = Path("tmp_upload")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zf)) as z:
        z.extractall(tmp_dir)
    return str(tmp_dir)

def list_images_labeled(root: str) -> List[Tuple[str, int]]:
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    items = []
    for idx, name in enumerate(["NORMAL", "PNEUMONIA"]):
        sub = Path(root)/name
        if not sub.exists():
            continue
        for p in sub.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                items.append((str(p), idx))
    return items

def predict_batch(model, items: List[Tuple[str,int]], img_size: int, batch:int=32):
    Xb, y_true, paths, probs = [], [], [], []
    for i, (path, lab) in enumerate(items, 1):
        pil = Image.open(path).convert("RGB")
        pil = crop_black_borders(pil)
        pil = pil.resize((img_size, img_size))
        arr = np.array(pil).astype(np.float32)/255.0
        Xb.append(arr); y_true.append(lab); paths.append(path)
        if len(Xb)==batch or i==len(items):
            X = np.stack(Xb,0)
            p = model.predict(X, verbose=0).reshape(-1)
            probs += p.tolist(); Xb = []
    return np.array(paths), np.array(y_true), np.array(probs)

def metrics_at_threshold(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"acc":acc, "precision":precision, "recall":recall, "f1":f1, "tp":tp, "tn":tn, "fp":fp, "fn":fn}

def choose_best_threshold(y_true, y_prob, metric="f1", min_recall=0.5):
    thrs = np.linspace(0.05, 0.95, 181)
    best = None
    for t in thrs:
        m = metrics_at_threshold(y_true, y_prob, t)
        if metric == "precision":
            if m["recall"] < min_recall:
                continue
            score = m["precision"]
        elif metric == "accuracy":
            score = m["acc"]
        elif metric == "recall":
            score = m["recall"]
        else:
            score = m["f1"]
        if (best is None) or (score > best[0]):
            best = (score, m, t)
    if best is None:
        return 0.5, metrics_at_threshold(y_true, y_prob, 0.5)
    return best[2], best[1]

def plot_roc_pr(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig1 = plt.figure(figsize=(4,3))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend()
    plt.tight_layout()

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig2 = plt.figure(figsize=(4,3))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.legend()
    plt.tight_layout()

    return fig1, fig2

def make_pdf_report(pil_img: Image.Image, overlay_cam: np.ndarray, overlay_campp: np.ndarray, prob: float, thr: float, mode: str, model_id: str = "densenet121_v1"):
    try:
        from fpdf import FPDF
    except Exception:
        return None  # fpdf2 not installed
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Pneumonia XR Prototype Report", ln=1, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Model: {model_id}", ln=1)
    pdf.cell(0, 7, f"Mode:  {mode} | Threshold: {thr:.2f}", ln=1)
    pdf.cell(0, 7, f"Prob(PNEUMONIA): {prob:.3f}", ln=1)

    # Save temporary images
    pil_path = "tmp_original.png"
    cam_path = "tmp_cam.png"
    campp_path = "tmp_campp.png"
    pil_img.save(pil_path)
    Image.fromarray(overlay_cam[:, :, ::-1]).save(cam_path)
    Image.fromarray(overlay_campp[:, :, ::-1]).save(campp_path)

    pdf.image(pil_path, x=15, y=50, w=80)
    pdf.image(cam_path, x=110, y=50, w=80)
    pdf.image(campp_path, x=110, y=140, w=80)
    pdf.set_xy(15, 140); pdf.cell(0, 7, "Original", ln=1)
    pdf.set_xy(110, 130); pdf.cell(0, 7, "Grad-CAM", ln=1)
    pdf.set_xy(110, 220); pdf.cell(0, 7, "Grad-CAM++", ln=1)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    # cleanup
    for p in [pil_path, cam_path, campp_path]:
        if os.path.exists(p): os.remove(p)
    return pdf_bytes

# -----------------------------
# Header
# -----------------------------
st.title("🫁 Pneumonia XR Prototype+")
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
# Main layout
# -----------------------------
colL, colR = st.columns([1,1])

with colL:
    uploaded = st.file_uploader("Sube una radiografía (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded is not None and model is not None:
    pil, img_norm = read_and_preprocess(uploaded, img_size, do_crop=enable_crop)

    # Image quality checks
    issues, p_dark, p_bright = image_quality_checks(pil, dark_cut, bright_cut)
    if issues:
        st.warning("⚠️ Posibles problemas de imagen: " + " | ".join(issues))
        st.caption(f"p_dark={p_dark*100:.1f}% · p_bright={p_bright*100:.1f}%")
        st.stop()

    target_layer = detect_target_layer(model)

    # Inference + CAMs
    t0 = time.time()
    cam, prob_cam = grad_cam(model, img_norm, target_layer)
    campp, prob_campp = grad_cam_pp(model, img_norm, target_layer)
    dt_ms = (time.time() - t0) * 1000

    # Overlays
    overlay_cam = make_overlay(cam, img_norm, alpha)
    overlay_campp = make_overlay(campp, img_norm, alpha)

    # Use default method to decide
    prob = prob_campp if method_default == "Grad-CAM++" else prob_cam
    decision = "PNEUMONIA" if prob >= threshold else "NORMAL"

    with colL:
        st.subheader("Imagen")
        st.image(pil, use_column_width=True)

        # Confidence bar
        st.subheader("Confianza")
        st.progress(int(prob * 100))
        nivel = "ALTO" if prob >= max(0.85, threshold+0.15) else ("INTERMEDIO" if prob >= threshold else "BAJO")
        st.caption(f"Riesgo: **{nivel}** · Prob={prob:.3f} · Umbral={threshold:.2f} · Modo={mode} · {dt_ms:.1f} ms")

    with colR:
        st.subheader("Resultado")
        st.metric("Prob(PNEUMONIA)", f"{prob:.3f}")
        st.write(f"**Decisión:** **{decision}**  (umbral {threshold:.2f})")

        tab1, tab2 = st.tabs(["Grad-CAM", "Grad-CAM++"])
        with tab1:
            st.image(overlay_cam, use_column_width=True, caption="Grad-CAM (regiones influyentes)")
        with tab2:
            st.image(overlay_campp, use_column_width=True, caption="Grad-CAM++ (regiones influyentes)")

        # Downloads
        st.download_button(
            "⬇️ Descargar overlay Grad-CAM (PNG)",
            data=cv2.imencode(".png", overlay_cam[:, :, ::-1])[1].tobytes(),
            file_name="gradcam_overlay.png",
            mime="image/png",
        )
        st.download_button(
            "⬇️ Descargar overlay Grad-CAM++ (PNG)",
            data=cv2.imencode(".png", overlay_campp[:, :, ::-1])[1].tobytes(),
            file_name="gradcampp_overlay.png",
            mime="image/png",
        )

        # Logging + JSON + PDF
        model_id = st.text_input("ID/Versión del modelo", value="densenet121_v1")
        log = {"ts": int(time.time()), "model": model_id, "prob": float(prob),
               "threshold": float(threshold), "mode": mode, "decision": decision}
        if st.button("Guardar en auditoría (CSV)"):
            csv_path = Path("audit.csv")
            df_log = pd.DataFrame([log])
            df_log.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
            st.success(f"Guardado en {csv_path}")

        st.download_button(
            "⬇️ Descargar resultado JSON",
            data=json.dumps(log, indent=2).encode("utf-8"),
            file_name="resultado.json",
            mime="application/json",
        )

        pdf_bytes = make_pdf_report(pil, overlay_cam, overlay_campp, prob, threshold, mode, model_id)
        if pdf_bytes is not None:
            st.download_button(
                "⬇️ Descargar reporte PDF",
                data=pdf_bytes,
                file_name="reporte_pneumonia.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Instala `fpdf2` para exportar PDF:  pip install fpdf2")

# -----------------------------
# Batch panel
# -----------------------------
st.markdown("---")
st.header("📦 Batch (ZIP con NORMAL/ y PNEUMONIA/)")
zip_file = st.file_uploader("Sube un .zip con carpetas NORMAL/ y PNEUMONIA/", type=["zip"], key="zipuploader")

if zip_file is not None and model is not None:
    with st.spinner("Procesando ZIP..."):
        root = extract_zip_to_temp(zip_file.getvalue())
        items = list_images_labeled(root)
        if not items:
            st.error("No se encontraron imágenes en NORMAL/ o PNEUMONIA/ dentro del ZIP.")
        else:
            paths, y_true, y_prob = predict_batch(model, items, img_size, batch=32)
            # Elegir mejor umbral por F1
            thr_f1, m_f1 = choose_best_threshold(y_true, y_prob, metric="f1")
            thr_prec, m_prec = choose_best_threshold(y_true, y_prob, metric="precision", min_recall=0.60)

            # Save CSV
            df = pd.DataFrame({
                "filepath": paths,
                "label": np.where(y_true==1,"PNEUMONIA","NORMAL"),
                "prob_pneumonia": y_prob,
                "pred@thr_f1": np.where(y_prob>=thr_f1,"PNEUMONIA","NORMAL"),
                "pred@thr_prec": np.where(y_prob>=thr_prec,"PNEUMONIA","NORMAL"),
            })
            csv_bytes = df.to_csv(index=False).encode("utf-8")

            # ROC/PR
            fig_roc, fig_pr = plot_roc_pr(y_true, y_prob)

            st.subheader("Resultados batch")
            st.write(f"Best F1 -> thr={thr_f1:.3f}  ·  Acc={m_f1['acc']:.3f}  Prec={m_f1['precision']:.3f}  Rec={m_f1['recall']:.3f}  F1={m_f1['f1']:.3f}")
            st.write(f"Max Prec (recall≥0.60) -> thr={thr_prec:.3f}  Prec={m_prec['precision']:.3f}  Rec={m_prec['recall']:.3f}")
            st.download_button("⬇️ Descargar results.csv", data=csv_bytes, file_name="results.csv", mime="text/csv")

            c1, c2 = st.columns(2)
            with c1: st.pyplot(fig_roc, use_container_width=True)
            with c2: st.pyplot(fig_pr, use_container_width=True)

            # Matriz de confusión para thr F1
            m = m_f1
            cm = np.array([[m["tn"], m["fp"]],[m["fn"], m["tp"]]], dtype=int)
            fig_cm = plt.figure(figsize=(4,3))
            plt.imshow(cm, cmap="Blues")
            plt.xticks([0,1], ["Pred NORMAL","Pred PNEUMONIA"])
            plt.yticks([0,1], ["True NORMAL","True PNEUMONIA"])
            for (i,j), val in np.ndenumerate(cm):
                plt.text(j, i, str(val), ha="center", va="center", fontsize=12)
            plt.title("Matriz de confusión (thr@F1)"); plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)

st.markdown("---")
st.caption("© Prototype+ • DenseNet121 • Preproc: gris→RGB, 224, /255.0 • CAM auto • Batch ZIP • Auditoría")
