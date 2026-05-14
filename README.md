# 🍎 Apple Inspector — Flask Web App

Aplikasi web Flask untuk mendeteksi kesegaran apel (Fresh / Rotten)
menggunakan model **EfficientNetB0** hasil transfer learning.

---

## 📁 Struktur Folder

```
apple_detector/
├── app.py                                  ← Flask backend
├── requirements.txt
├── efficientnetb0_fresh_rotten_apple.keras ← letakkan di sini
├── class_names.json                        ← opsional
├── templates/
│   └── index.html
└── uploads/                                ← temp (otomatis dibuat)
```

---

## 🚀 Cara Menjalankan

### 1. Clone / copy folder ini, lalu masuk ke dalamnya
```bash
cd apple_detector
```

### 2. Buat virtual environment (disarankan)
```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Letakkan file model
Salin `efficientnetb0_fresh_rotten_apple.keras` ke folder yang sama dengan `app.py`.

### 5. Jalankan Flask
```bash
python app.py
```

Buka browser → **http://localhost:5000**

---

## ⚙️ Konfigurasi

| Variabel lingkungan | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `efficientnetb0_fresh_rotten_apple.keras` | Path ke file model `.keras` |

Contoh:
```bash
MODEL_PATH=/data/model.keras python app.py
```

---

## 🔌 API Endpoint

### `POST /predict`
Upload gambar apel, dapatkan prediksi JSON.

**Request:** `multipart/form-data`, field `image` (jpg/png/webp, maks 10 MB)

**Response:**
```json
{
  "label": "Fresh",
  "confidence": 97.43,
  "preview": "data:image/jpeg;base64,..."
}
```

### `GET /health`
Cek status server & model.

```json
{ "status": "ok", "model_loaded": true }
```

---

## ⚠️ Catatan Kekurangan Model (lihat analisis lengkap di README)

1. **Tidak ada fine-tuning** — base EfficientNetB0 di-freeze seluruhnya (10 epoch frozen-only).
2. **Tidak ada early stopping** — training berjalan penuh 10 epoch tanpa `EarlyStopping` atau `ModelCheckpoint`.
3. **Augmentasi terbatas** — hanya Flip, Rotation, Zoom, Contrast; tidak ada brightness/hue shift.
4. **Satu Dense layer** — head klasifikasi hanya `Dropout → Dense(1, sigmoid)`, tanpa hidden layer tambahan.
5. **Learning rate tetap** — tidak ada `ReduceLROnPlateau` atau scheduler.
6. **Label_mode binary** — urutan kelas bergantung pada urutan folder alfabetis (Fresh < Rotten) yang tidak eksplisit dicatat dalam kode.
