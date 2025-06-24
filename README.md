# FaceSwap-Pro-Video-Audio-Subtitle-Integration
FaceSwap-Pro is a comprehensive toolkit for **face swapping on images and videos**, with full audio support and two subtitle options: **standard subtitles** or **dynamic, real-time highlighted subtitles**.
---

## ✨ Features

- ✅ **Face Swapping on Images & Videos**
- ✅ **Original Audio Retention** in swapped videos
- ✅ **Subtitles Generation** using OpenAI Whisper
- ✅ **Two Subtitle Modes**:
  - 🔸 **Standard Subtitles** (using ffmpeg, professional look)
  - 🔸 **Dynamic Highlighted Subtitles** (custom OpenCV render, word-by-word highlighting)

---

## 📂 Folder Structure
```

FaceSwap-Pro/
├── models/        # Store model files here (download manually)
├── scripts/       # All working scripts
├── examples/      # Example images/videos
└── README.md

````

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/FaceSwap-Pro.git
cd FaceSwap-Pro
pip install -r requirements.txt
````

Install **ffmpeg** separately:

```bash
sudo apt-get install ffmpeg
```

### ⚠️ Download Model

* Download `inswapper_128.onnx` 
* Place it in `models/` directory.

---

## 🚀 Usage

### 📷 Swap Face in an Image

```bash
python scripts/image_swap.py
```

### 🎬 Swap Face in Video (With Original Audio)

```bash
python scripts/face_swap_video.py
```

### 📝 Full Pipeline (Face Swap + Subtitles burned in)

```bash
python scripts/full.py
```

### 🖍 Dynamic Highlighted Subtitles (Custom Visualization)

```bash
python scripts/Full_pipeline.py
```

> ⚠ Dynamic mode may be slower but demonstrates advanced real-time caption rendering.

---

## 📝 Requirements

* Python 3.8+
* ffmpeg (for audio handling and subtitle rendering)
* Required Python packages: (see `requirements.txt`)

---
