from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os, io, base64, threading
from PIL import Image

app = Flask(__name__, static_folder="static")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH        = os.environ.get("MODEL_PATH", "best_cnn_model.h5")
ESRGAN_URL        = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
ESRGAN_CACHE_PATH = "esrgan_saved_model"   # written once by cache_esrgan.py

TILE    = 128
OVERLAP = 16

# ── Lazy loaders ──────────────────────────────────────────────────────────────
_cnn_model    = None
_esrgan_model = None

def get_cnn():
    global _cnn_model
    if _cnn_model is None:
        import tensorflow as tf
        def psnr_metric(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)
        _cnn_model = tf.keras.models.load_model(
            MODEL_PATH, custom_objects={"psnr_metric": psnr_metric}
        )
        print("[INFO] CNN loaded.")
    return _cnn_model

def get_esrgan():
    global _esrgan_model
    if _esrgan_model is None:
        import tensorflow as tf
        if os.path.exists(ESRGAN_CACHE_PATH):
            # ✅ Fast — load from local disk, no internet needed
            print(f"[INFO] Loading ESRGAN from local cache...")
            _esrgan_model = tf.saved_model.load(ESRGAN_CACHE_PATH)
            print("[INFO] ESRGAN loaded from cache.")
        else:
            # ⬇ Fallback — download from TF Hub (slow, first time only)
            import tensorflow_hub as hub
            print("[WARN] Cache not found. Downloading ESRGAN from TF Hub...")
            print("[WARN] Run 'python cache_esrgan.py' once to cache it locally.")
            _esrgan_model = hub.load(ESRGAN_URL)
            print("[INFO] ESRGAN loaded from TF Hub.")
    return _esrgan_model

def warmup_esrgan():
    """
    Runs a dummy 32x32 inference in a background thread at startup.
    This pre-compiles the TF graph so the first real request is instant.
    """
    def _run():
        import tensorflow as tf
        try:
            print("[INFO] Warming up ESRGAN in background...")
            sr    = get_esrgan()
            dummy = tf.constant(np.zeros((1, 32, 32, 3), dtype=np.float32))
            _     = sr(dummy)
            print("[INFO] ESRGAN warm-up complete. First request will be fast.")
        except Exception as e:
            print(f"[WARN] Warm-up failed: {e}")
    threading.Thread(target=_run, daemon=True).start()

# ── Helpers ───────────────────────────────────────────────────────────────────
def img_to_b64(img: Image.Image, fmt="JPEG", quality=92) -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

# ── Step 1: Tiled CNN enhancement ─────────────────────────────────────────────
def cnn_enhance(pil_img: Image.Image) -> Image.Image:
    m   = get_cnn()
    img = pil_img.convert("RGB")
    W, H = img.size

    if W < TILE or H < TILE:
        arr  = np.array(img.resize((TILE, TILE)), dtype=np.float32) / 255.0
        pred = m.predict(arr[np.newaxis], verbose=0)[0]
        out  = (np.clip(pred, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(out).resize((W, H), Image.LANCZOS)

    arr    = np.array(img, dtype=np.float32) / 255.0
    canvas = np.zeros_like(arr)
    weight = np.zeros((H, W, 1), dtype=np.float32)
    step   = TILE - OVERLAP

    ys = list(range(0, H - TILE, step)) + [H - TILE]
    xs = list(range(0, W - TILE, step)) + [W - TILE]

    for y in ys:
        for x in xs:
            patch = arr[y:y+TILE, x:x+TILE]
            pred  = m.predict(patch[np.newaxis], verbose=0)[0]
            pred  = np.clip(pred, 0, 1)

            w    = np.ones((TILE, TILE, 1), dtype=np.float32)
            ramp = np.linspace(0, 1, OVERLAP)
            if x > 0:        w[:, :OVERLAP]  *= ramp[np.newaxis, :, np.newaxis]
            if x+TILE < W:   w[:, -OVERLAP:] *= ramp[::-1][np.newaxis, :, np.newaxis]
            if y > 0:        w[:OVERLAP]      *= ramp[:, np.newaxis, np.newaxis]
            if y+TILE < H:   w[-OVERLAP:]     *= ramp[::-1][:, np.newaxis, np.newaxis]

            canvas[y:y+TILE, x:x+TILE] += pred * w
            weight[y:y+TILE, x:x+TILE] += w

    canvas = canvas / np.maximum(weight, 1e-6)
    out    = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out)

# ── Step 2: ESRGAN 4x upscale ─────────────────────────────────────────────────
MAX_ESRGAN_DIM = 300

def esrgan_upscale(pil_img: Image.Image) -> Image.Image:
    import tensorflow as tf
    sr  = get_esrgan()
    img = pil_img.convert("RGB")
    W, H = img.size

    if W > MAX_ESRGAN_DIM or H > MAX_ESRGAN_DIM:
        scale = MAX_ESRGAN_DIM / max(W, H)
        img   = img.resize((int(W*scale), int(H*scale)), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    t   = tf.convert_to_tensor(arr[np.newaxis])
    out = sr(t)
    out = tf.clip_by_value(out, 0, 255)
    out = tf.cast(out, tf.uint8).numpy()[0]
    return Image.fromarray(out)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/enhance", methods=["POST"])
def enhance():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    mode = request.form.get("mode", "hd")

    try:
        pil_img  = Image.open(request.files["image"].stream)
        original = pil_img.convert("RGB")

        print("[INFO] Step 1: CNN enhancement...")
        enhanced = cnn_enhance(original)

        hd_b64 = None
        if mode == "hd":
            print("[INFO] Step 2: ESRGAN upscale...")
            hd     = esrgan_upscale(enhanced)
            hd_b64 = img_to_b64(hd)
            print(f"[INFO] HD output: {hd.size}")

        return jsonify({
            "original" : img_to_b64(original),
            "enhanced" : img_to_b64(enhanced),
            "hd"       : hd_b64,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
    warmup_esrgan()   # pre-compile graph in background while Flask starts
    app.run(debug=False, port=5000)
