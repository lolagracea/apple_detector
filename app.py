import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'

# ── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get('MODEL_PATH', 'efficientnetb0_fresh_rotten_apple.keras')

model = None
class_names = ['Fresh', 'Rotten']   # default fallback

def load_model():
    global model, class_names
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"[✗] Could not load model: {e}")
        model = None

    # try loading class names if saved alongside model
    if os.path.exists('class_names.json'):
        with open('class_names.json') as f:
            class_names = json.load(f)
        print(f"[✓] class_names: {class_names}")

load_model()

# ── Preprocessing ────────────────────────────────────────────────────────────
IMG_SIZE = (224, 224)

def preprocess_image(pil_image: Image.Image):
    import tensorflow as tf
    img = pil_image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)   # (1, 224, 224, 3)

def predict(pil_image: Image.Image):
    if model is None:
        return None, None, "Model belum dimuat."
    tensor = preprocess_image(pil_image)
    prob = float(model.predict(tensor, verbose=0)[0][0])
    # label_mode='binary' → index 0 = Fresh, index 1 = Rotten
    label_idx = int(prob > 0.5)
    label = class_names[label_idx] if label_idx < len(class_names) else ('Rotten' if label_idx else 'Fresh')
    confidence = prob if label_idx == 1 else 1 - prob
    return label, round(confidence * 100, 2), None

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Format tidak didukung: {ext}'}), 400

    try:
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({'error': 'Gambar tidak valid.'}), 400

    label, confidence, err = predict(pil_image)
    if err:
        return jsonify({'error': err}), 500

    # Return thumbnail for preview
    pil_image.thumbnail((400, 400))
    buf = io.BytesIO()
    pil_image.save(buf, format='JPEG', quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        'label': label,
        'confidence': confidence,
        'preview': f'data:image/jpeg;base64,{b64}'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
