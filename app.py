import atexit

from flask import Flask, jsonify, render_template, Response, url_for, request, redirect
from backend.wrapper import Wrapper

app = Flask(__name__)
wrapper = Wrapper()
atexit.register(wrapper._capture.stop)
selected_camera = None
camera_supplier = lambda: selected_camera

@app.route("/")
def camera():
    global cameras, selected_camera
    cameras = wrapper.get_cameras()
    if selected_camera not in cameras:
        selected_camera = cameras[0] if cameras else None
    return render_template("camera.html", cameras=cameras, selected_index=selected_camera)

@app.route("/set_camera", methods=["POST"])
def set_camera():
    global selected_camera
    cam_name = request.form.get("camera")
    if cam_name in cameras:
        selected_camera = cam_name
    return redirect(url_for("camera"))

@app.route('/video_feed')
def video_feed():
    return Response(wrapper.get_raw_frame(camera_supplier), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rename_camera', methods=['POST'])
def rename_camera_route():
    data = request.get_json()
    index = int(data.get('index'))
    new_name: str = data.get('name', '').strip()

    if new_name and wrapper.update_config(index, 'camera_name', new_name):
        global selected_camera
        selected_camera = new_name  # Update selected camera to the renamed one
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/get_camera_settings')
def get_camera_settings():
    index = int(request.args.get('index'))
    settings = wrapper.get_camera_settings(index)
    return jsonify(settings)

@app.route('/set_camera_setting', methods=['POST'])
def set_camera_setting():
    data = request.get_json()
    index = int(data.get('index'))
    key = data.get('key')
    value = data.get('value')

    success = wrapper.update_config(index, key, value)
    return jsonify(success=success)

@app.route("/calibration")
def calibration():
    return render_template("calibration.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)