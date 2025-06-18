'''
[ 摄像头/视频流 ]
        │
        ▼
[ 视频帧提取 (OpenCV) ]
        │
        ▼
[ YOLOv11 实时检测 ]
        │
        ▼
[ 剩余面积分析 + 判定逻辑 ]
        │
        ▼
[ 显示 + 存储结果 ]
        │
        └──> 警告/统计（接口输出）

'''

 
# 安装 Python 和依赖
brew install python3
python3 -m venv yolov11
source yolov11/bin/activate

pip install ultralytics opencv-python inference


	•	ultralytics 含 YOLO11 模型支持  ￼ ￼。
	•	inference 可用于 macOS 本地推理 。

python detect_stream_mac.py