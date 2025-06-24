import cv2
from deepface import DeepFace
from recognize_face import recg_face
import os

# 模拟 name_map.json
# name_map = {
#     "person1": "张三",
#     "person2": "李四",
#     "jiangyang": "姜阳"
# }

# 测试入口
if __name__ == "__main__":
    # 或者相对路径（相对当前脚本）
    test_img_path = os.path.join(os.path.dirname(__file__), "test_face.jpg")
    # 替换为你本地的一张待识别人脸照片路径
    test_face_path = "test_face.jpg"

    if not os.path.exists(test_face_path):
        print(f"[ERROR] 测试图片 {test_face_path} 不存在，请确认路径正确")
    else:
        result = recg_face(test_face_path)
        print(f"[RESULT] 识别结果：{result}")