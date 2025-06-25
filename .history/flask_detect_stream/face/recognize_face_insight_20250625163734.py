import os
import cv2
import numpy as np
import pickle
import insightface

# 初始化模型
model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0, det_size=(320, 320))  # CPU=-1, GPU=0

# 注册人脸入库
def register_face(img_path: str, name: str, known_faces: dict):
    img = cv2.imread(img_path)
    faces = model.get(img)
    if not faces:
        print(f"[WARN] 未识别到人脸：{img_path}")
        return
    vec = faces[0].embedding
    known_faces.setdefault(name, []).append(vec)
    print(f"[INFO] 人脸已录入：{name}")

# 识别人脸
def fast_face_recognize(image: np.ndarray, known_faces: dict, threshold: float = 1.0) -> str:
    faces = model.get(image)
    if not faces:
        print("[INFO] 未识别到人脸")
        return "未知"
    query_vec = faces[0].embedding
    best_dist = float("inf")
    best_name = "未知"
    for name, emb_list in known_faces.items():
        for ref in emb_list:
            dist = np.linalg.norm(query_vec - ref)
            if dist < best_dist:
                best_dist = dist
                best_name = name
    if best_dist < threshold:
        print(f"[INFO] 识别为：{best_name}（距离：{best_dist:.2f}）")
        return best_name
    print(f"[INFO] 未识别成功（最小距离：{best_dist:.2f}）")
    return "未知"

# 保存人脸库
def save_face_lib(known_faces: dict, path: str = "face_lib.pkl"):
    with open(path, "wb") as f:
        pickle.dump(known_faces, f)
    print(f"[INFO] 已保存人脸库到：{path}")

# 加载人脸库
def load_face_lib(path: str = "face_lib.pkl") -> dict:
    if not os.path.exists(path):
        print("[WARN] 人脸库文件不存在，初始化空库")
        return {}
    with open(path, "rb") as f:
        print(f"[INFO] 成功加载人脸库：{path}")
        return pickle.load(f)

# 用法示例（可在主函数或外部调用）
if __name__ == "__main__":
    known_faces = load_face_lib()
    register_face("张三.jpg", "张三", known_faces)
    register_face("李四.jpg", "李四", known_faces)
    save_face_lib(known_faces)

    test_img = cv2.imread("test.jpg")
    result = fast_face_recognize(test_img, known_faces)
    print("识别结果：", result)