"""
Tomato Leaf Disease Detection — Flask Backend
LOCAL MAC VERSION — paths set for your machine
Run: python app.py
API: http://localhost:5000
"""

import os, io, json, base64, logging, time
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import Model
import matplotlib.cm as cm

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ── Paths — resolve from this file so the project works on any machine ───────
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / 'models' / 'tomato_disease_vgg16.h5'
LABELS_PATH = BASE_DIR / 'models' / 'class_labels.json'
IMAGE_SIZE  = (224, 224)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp', 'bmp'}
CONFIDENCE_FLOOR   = 35.0   # below this % → flag as "Not Tomato Leaf"

# ── Disease info ──────────────────────────────────────────────────────────────
DISEASE_INFO = {
    'Tomato___Bacterial_spot': {
        'description': 'Dark, water-soaked lesions on leaves and fruit. Caused by Xanthomonas vesicatoria.',
        'treatment'  : 'Apply copper-based bactericides. Remove infected plant debris. Use disease-free seeds.',
        'severity'   : 'Moderate',
    },
    'Tomato___Early_blight': {
        'description': 'Concentric ring lesions with yellow halo. Caused by Alternaria solani.',
        'treatment'  : 'Apply fungicides (chlorothalonil, mancozeb). Improve air circulation.',
        'severity'   : 'Moderate',
    },
    'Tomato___Late_blight': {
        'description': 'Water-soaked oily spots that turn brown. Caused by Phytophthora infestans.',
        'treatment'  : 'Apply fungicides immediately. Remove infected tissue. Avoid wet conditions.',
        'severity'   : 'Severe',
    },
    'Tomato___Leaf_Mold': {
        'description': 'Yellow spots on upper leaf surface, olive-grey mold on lower surface.',
        'treatment'  : 'Increase ventilation. Reduce humidity. Apply fungicides.',
        'severity'   : 'Moderate',
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Small circular spots with dark borders and grey centres.',
        'treatment'  : 'Remove infected leaves. Apply fungicides. Practice crop rotation.',
        'severity'   : 'Moderate',
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Tiny mites causing stippled pale leaves and fine webbing.',
        'treatment'  : 'Use miticides or insecticidal soap. Increase humidity.',
        'severity'   : 'Moderate',
    },
    'Tomato___Target_Spot': {
        'description': 'Circular spots with concentric rings. Caused by Corynespora cassiicola.',
        'treatment'  : 'Apply fungicides. Remove infected leaves. Improve air circulation.',
        'severity'   : 'Low to Moderate',
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Yellowing and curling of leaves. Transmitted by whiteflies.',
        'treatment'  : 'Control whitefly vectors. Use resistant varieties. Remove infected plants.',
        'severity'   : 'Severe',
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'Mosaic pattern of light and dark green areas on leaves.',
        'treatment'  : 'No cure. Remove infected plants. Disinfect tools. Use resistant varieties.',
        'severity'   : 'Severe',
    },
    'Tomato___healthy': {
        'description': 'The tomato leaf appears healthy with no visible disease symptoms.',
        'treatment'  : 'No treatment required. Continue good agricultural practices.',
        'severity'   : 'None',
    },
}

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
FRONTEND_DIR = BASE_DIR / 'frontend'
CORS(app, origins='*')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB

# ── Load model at startup ─────────────────────────────────────────────────────
model      = None
idx_to_cls = None
cls_to_idx = None

def load_artifacts():
    global model, idx_to_cls, cls_to_idx
    if not MODEL_PATH.exists():
        log.warning(f'Model not found: {MODEL_PATH}')
        log.warning('Run train_local.py first to generate the model.')
        return False
    log.info(f'Loading model from {MODEL_PATH} …')
    model = load_model(str(MODEL_PATH))
    with open(LABELS_PATH) as f:
        data = json.load(f)
    cls_to_idx = data['class_indices']
    idx_to_cls = {int(k): v for k, v in data['idx_to_class'].items()}
    log.info(f'Model loaded successfully. Classes: {len(idx_to_cls)}')
    return True

artifacts_loaded = load_artifacts()

# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(pil_img):
    img = pil_img.convert('RGB').resize(IMAGE_SIZE)
    arr = np.array(img, dtype='float32')
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def is_tomato_leaf_like(pil_img):
    img  = pil_img.convert('RGB').resize((64, 64))
    arr  = np.array(img, dtype='float32')
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    green_mask  = (g > r) & (g > b) & (g > 50)
    return bool(green_mask.mean() > 0.05)


def make_gradcam_heatmap(img_array):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer('block5_conv3').output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        pred_idx        = tf.argmax(preds[0])
        class_score     = preds[:, pred_idx]
    grads   = tape.gradient(class_score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_gradcam(orig_rgb, heatmap, alpha=0.45):
    jet        = cm.get_cmap('jet')
    colored    = np.uint8(jet(heatmap)[:, :, :3] * 255)
    colored_pil = Image.fromarray(colored).resize(
        (orig_rgb.shape[1], orig_rgb.shape[0]), Image.LANCZOS
    )
    return (np.array(colored_pil) * alpha + orig_rgb * (1 - alpha)).astype(np.uint8)

def pil_to_base64(pil_img, fmt='JPEG'):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════
@app.route('/', methods=['GET'])
def home():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'       : 'ok',
        'model_loaded' : artifacts_loaded,
        'num_classes'  : len(idx_to_cls) if idx_to_cls else 0,
        'tensorflow'   : tf.__version__,
    })


@app.route('/classes', methods=['GET'])
def classes():
    if not artifacts_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({'classes': [
        {
            'index' : idx,
            'name'  : name,
            'short' : name.replace('Tomato___', '').replace('_', ' '),
            'info'  : DISEASE_INFO.get(name, {}),
        }
        for idx, name in idx_to_cls.items()
    ]})


@app.route('/predict', methods=['POST'])
def predict():
    if not artifacts_loaded:
        return jsonify({'error': 'Model not loaded. Run train_local.py first.'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    t0 = time.time()
    try:
        pil_img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Cannot open image: {e}'}), 400

    # Thumbnail for display
    thumb = pil_img.copy()
    thumb.thumbnail((400, 400))
    thumb_b64 = pil_to_base64(thumb)

    try:
        arr   = preprocess_image(pil_img)
        probs = model.predict(arr, verbose=0)[0]
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    top_indices = np.argsort(probs)[::-1][:3]
    top_conf    = float(probs[top_indices[0]]) * 100
    is_tomato = bool((top_conf >= CONFIDENCE_FLOOR) and is_tomato_leaf_like(pil_img))

    def make_pred(idx):
        name = idx_to_cls[int(idx)]
        return {
            'class'      : name,
            'short'      : name.replace('Tomato___', '').replace('_', ' '),
            'confidence' : round(float(probs[idx]) * 100, 2),
            'info'       : DISEASE_INFO.get(name, {})
        }

    top3       = [make_pred(i) for i in top_indices]
    latency_ms = round((time.time() - t0) * 1000, 1)

    return jsonify({
        'is_tomato'      : is_tomato,
        'top_prediction' : top3[0] if is_tomato else {
            'class'      : 'Not_Tomato_Leaf',
            'short'      : 'Not a Tomato Leaf',
            'confidence' : round(top_conf, 2),
            'info'       : {
                'description': 'The image does not appear to be a tomato leaf.',
                'treatment'  : 'Please upload a clear image of a tomato leaf.',
                'severity'   : 'N/A',
            }
        },
        'top3'         : top3,
        'thumb_base64' : thumb_b64,
        'latency_ms'   : latency_ms,
    })


@app.route('/gradcam', methods=['POST'])
def gradcam():
    if not artifacts_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    if 'image' not in request.files:
        return jsonify({'error': 'No image field'}), 400

    try:
        pil_img      = Image.open(io.BytesIO(request.files['image'].read())).convert('RGB')
        orig_resized = np.array(pil_img.resize(IMAGE_SIZE))
        arr          = preprocess_input(np.expand_dims(orig_resized.astype('float32'), 0))
        heatmap      = make_gradcam_heatmap(arr)
        heatmap_full = np.array(
            Image.fromarray(np.uint8(heatmap * 255)).resize(IMAGE_SIZE, Image.LANCZOS)
        ) / 255.0
        superimposed = overlay_gradcam(orig_resized, heatmap_full)
        return jsonify({'gradcam_base64': pil_to_base64(Image.fromarray(superimposed))})
    except Exception as e:
        return jsonify({'error': f'Grad-CAM failed: {e}'}), 500


if __name__ == '__main__':
    print('\n' + '='*55)
    print(' TomatoScan AI — Flask Backend')
    print(f' Model path : {MODEL_PATH}')
    print(f' API        : http://localhost:5000')
    print('='*55 + '\n')
    app.run(host='0.0.0.0', port=5000, debug=False)
