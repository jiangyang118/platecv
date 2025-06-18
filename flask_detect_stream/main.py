
from flask import Flask, render_template, Response
from app.detect_stream_mac import run_detection  # 你需要将 detect_stream_mac.py 中封装成 run_detection 函数

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(run_detection(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)
