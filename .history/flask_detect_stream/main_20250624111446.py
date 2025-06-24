
from flask import Flask, render_template, Response
from app.detect_stream_mac import run_detection  # 你需要将 detect_stream_mac.py 中封装成 run_detection 函数
from app.detect_stream_face import run_detection_face_food  # 如果需要人脸识别和食物检测，可以使用这个函数


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # source = 'https://mclz-yy-sz.esunego.com/mclz-yy-sz/43060300501327000005.m3u8'
    # source = 'https://open.ys7.com/v3/openlive/FB8747881_1_2.m3u8?expire=1781404965&id=855759506062184448&t=3e69e9b3484ebf53fa2508a41e92d19e7fbe53ae539ca5095002a834660c2351&ev=100'
    # source = 'https://open.ys7.com/v3/openlive/FB8747881_1_1.m3u8?expire=1781334713&id=855464846889951232&t=8b6f20ef6cefdf456992988492969c65b1a73d7d9590fbe9b36d99122703a5dc&ev=100'
    source = 0
    return Response(run_detection(source), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
