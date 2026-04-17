"""
Run this ONCE to download ESRGAN and save it locally:
    python cache_esrgan.py

After this, app.py will load ESRGAN from disk (fast) instead of
downloading it from TF Hub every time.
"""

import os
import tensorflow as tf
import tensorflow_hub as hub

ESRGAN_URL        = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
ESRGAN_CACHE_PATH = "esrgan_saved_model"

def main():
    if os.path.exists(ESRGAN_CACHE_PATH):
        print(f"[INFO] Cache already exists at '{ESRGAN_CACHE_PATH}'. Nothing to do.")
        print("[INFO] Delete the folder and re-run if you want to re-download.")
        return

    print("[INFO] Downloading ESRGAN from TF Hub...")
    print("[INFO] This only happens once (~5 MB download).")
    model = hub.load(ESRGAN_URL)

    print(f"[INFO] Saving to '{ESRGAN_CACHE_PATH}'...")
    tf.saved_model.save(model, ESRGAN_CACHE_PATH)

    # Quick warm-up inference to compile the graph into the SavedModel
    print("[INFO] Running warm-up inference to bake the compute graph...")
    import numpy as np
    dummy = tf.constant(np.zeros((1, 50, 50, 3), dtype=np.float32))
    _ = model(dummy)
    print("[INFO] Done! ESRGAN is cached locally.")
    print(f"[INFO] Folder size: {get_dir_size(ESRGAN_CACHE_PATH):.1f} MB")

def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)

if __name__ == "__main__":
    main()
