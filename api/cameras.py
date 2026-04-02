from flask import Blueprint, Response, jsonify
from backend.managers.camera_manager import CameraManager

bp = Blueprint("cameras", __name__, url_prefix="/api/cameras")

manager = CameraManager()

def generate(camera):
    while True:
        frame = camera.get_jpeg()
        if frame is None:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@bp.route("/stream/<name>")
def stream(name):
    cam = manager.get_camera(name)
    return Response(generate(cam),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@bp.route("/fps/<name>")
def fps(name):
    return jsonify({"fps": manager.get_fps(name)})