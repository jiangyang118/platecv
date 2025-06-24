# prepare_face_db.py
# 训练人脸数据库
import os
from deepface import DeepFace
import pickle

face_embeddings = {}
for person in os.listdir('face_db'):
    embeddings = []
    for f in os.listdir(f'face_db/{person}'):
        emb = DeepFace.represent(img_path=f'face_db/{person}/{f}', model_name='ArcFace', enforce_detection=True)
        embeddings.append(emb[0]["embedding"])
    face_embeddings[person] = embeddings

with open('face_embeddings.pkl', 'wb') as f:
    pickle.dump(face_embeddings, f)