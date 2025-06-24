import os
from deepface import DeepFace


with open('./name_map.json', 'r', encoding='utf‑8') as f:
    name_map = __import__('json').load(f)
 

file_path = os.path.join(os.path.dirname(__file__), "name_map.json")
with open(file_path, "r", encoding="utf-8") as f:
    name_map = json.load(f)
# 人脸识别
def recg_face(face_img):
    dfs = DeepFace.find(img_path=face_img, db_path='./face_db',
                       model_name='ArcFace', enforce_detection=False)
    print(f"[DEBUG] Face recognition results: {dfs}")
    if not dfs or dfs[0].empty:
        return "未知"
    eng = dfs[0].iloc[0]["identity"].split("/")[-2]
    return name_map.get(eng, "未知")
