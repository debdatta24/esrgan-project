# LumaLift — Low-Light Image Enhancer

A Flask web app that uses your trained `best_cnn_model.h5` to enhance low-light photos in the browser.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your model
Copy `best_cnn_model.h5` into this folder (next to `app.py`).

### 3. Run the server
```bash
python app.py
```

### 4. Open in browser
Go to `http://localhost:5000`

---

## Project Structure
```
low_light_app/
├── app.py              ← Flask backend
├── requirements.txt
├── best_cnn_model.h5   ← your trained model (add this)
└── static/
    └── index.html      ← frontend UI
```

## How It Works
- You drag/drop or select a low-light image in the browser.
- The image is sent to `/enhance` (POST).
- Flask resizes it to 128×128, runs it through the CNN, and returns both the original and enhanced images as base64.
- The frontend shows them side by side with a download button.

## Model Details
- **Architecture**: Residual CNN (Conv2D × 5 + skip connection)
- **Input**: 128 × 128 × 3 (normalized to [0, 1])
- **Output**: 128 × 128 × 3 (sigmoid activation)
- **Custom metric**: `psnr_metric` (loaded automatically)

## Environment Variable
You can point to a different model path:
```bash
MODEL_PATH=/path/to/model.h5 python app.py
```
