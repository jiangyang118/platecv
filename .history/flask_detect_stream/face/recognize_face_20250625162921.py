import os
import json
import pickle
import time
from functools import lru_cache

import cv2
import numpy as np
from deepface import DeepFace

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "face_db")
EMB_PATH = os.path.join(BASE_DIR, "face_embeddings.pkl")
NAME_MAP_PATH = os.path.join(BASE_DIR, "name_map.json")

with open(NAME_MAP_PATH, "r", encoding="utf-8") as f:
    NAME_MAP = json.load(f)

if os.path.exists(EMB_PATH):
    with open(EMB_PATH, "rb") as f:
        FACE_EMBEDDINGS = pickle.load(f)
else:
    FACE_EMBEDDINGS = {}

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = DeepFace.build_model("ArcFace")
    return _MODEL

def _embed(face_img: np.ndarray) -> np.ndarray:
    model = _get_model()
    rep = DeepFace.represent(face_img, model_name="ArcFace", model_name=model, enforce_detection=False)[0]
    return np.array(rep["embedding"], dtype=np.float32)


def recg_face_fast(face_img: np.ndarray, threshold: float = 45.0) -> str:
    """Return the recognized name using pre-computed embeddings."""
    embedding = _embed(face_img)
    best_name = "未知"
    best_dist = float("inf")
    for name, embs in FACE_EMBEDDINGS.items():
        for e in embs:
            dist = np.linalg.norm(embedding - np.array(e, dtype=np.float32))
            if dist < best_dist:
                best_dist = dist
                best_name = name
    if best_dist < threshold:
        print(f"[INFO] Recognized face: {best_name} with distance {best_dist:.2f}")
        return NAME_MAP.get(best_name, best_name)
    print(f"[INFO] Face not recognized, distance {best_dist:.2f}")
    return "未知"


def recg_face(image_path: str) -> str:
    """Compatibility wrapper using DeepFace.find."""
    dfs = DeepFace.find(img_path=image_path, db_path=DB_PATH, model_name="ArcFace", enforce_detection=False)
    if not dfs or dfs[0].empty:
        return "未知"
    eng = dfs[0].iloc[0]["identity"].split("/")[-2]
    return NAME_MAP.get(eng, "未知")


def save_face(face_imgs) -> None:
    os.makedirs("extracted_faces", exist_ok=True)
    for i, face in enumerate(face_imgs):
        face_rgb = face["face"]
        face_rgb_uint8 = (face_rgb * 255).astype("uint8")
        face_bgr = cv2.cvtColor(face_rgb_uint8, cv2.COLOR_RGB2BGR)
        timestamp = int(time.time())
        save_path = f"extracted_faces/face_{timestamp}_{i}.jpg"
        cv2.imwrite(save_path, face_bgr)
        print(f"[INFO] Saved face image to {save_path}")

