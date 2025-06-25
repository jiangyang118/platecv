# PlateCV

This project provides a simple demonstration of food waste detection using YOLO models and optional face recognition. It includes a command line demo and a small Flask application for streaming video.

## Setup

1. Install Python 3.9 or later.
2. Create a virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

Models used by the demo are already placed under `models/` and `flask_detect_stream/models/`.

## Quick Start

### Run YOLO detection from the command line

```bash
python detect_stream_mac.py
```

### Start the Flask stream server

```bash
python flask_detect_stream/main.py
```

Open your browser at `http://127.0.0.1:5000/` to view the stream.

## Preparing the Face Database

If you wish to use face recognition, place subfolders of face images under `flask_detect_stream/face/face_db/` and run:

```bash
python flask_detect_stream/face/prepare_face_db.py
```

This will generate `face_embeddings.pkl` used for fast recognition. The optimized pipeline uses these pre-computed embeddings for better performance.
