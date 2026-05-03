import atexit
import os
import select
import threading
import time

from ServerThread import ServerThread
from flask import Flask, jsonify, render_template, Response, url_for, request, redirect
from backend.wrapper import Wrapper

app = Flask(__name__)
wrapper = Wrapper()
atexit.register(wrapper._capture.stop)
selected_camera = None
camera_supplier = lambda: selected_camera
init = False

def initialize():
    global init, cameras, selected_camera
    if not init:
        cameras = wrapper.get_cameras()
        selected_camera = cameras[0] if cameras else None
        
        for i in range(len(cameras)):
            threading.Thread(target=wrapper.start_backend, args=(i,), daemon=True).start()
        init = True

@app.route("/")
def camera():
    global cameras, selected_camera
    return render_template("camera.html", cameras=cameras, selected_cam=selected_camera)

@app.route("/set_camera", methods=["POST"])
def set_camera():
    global selected_camera
    cam_name = request.form.get("camera")
    if cam_name in cameras:
        selected_camera = cam_name
    return redirect(url_for("camera"))

@app.route('/video_feed')
def video_feed():
    return Response(wrapper.get_frame(camera_supplier), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rename_camera', methods=['POST'])
def rename_camera_route():
    data = request.get_json()
    index = int(data.get('index'))
    new_name: str = data.get('name', '').strip()

    if new_name and wrapper.update_config(index, 'camera_name', new_name):
        global cameras, selected_camera
        selected_camera = new_name  # Update selected camera to the renamed one
        cameras[index] = new_name
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
    if data.get('index') == '':
        return jsonify(success=False, message="No Cameras")
    index = int(data.get('index'))
    key = data.get('key')
    value = data.get('value')

    success = wrapper.update_config(index, key, value)
    return jsonify(success=success)

@app.route('/set_camera_resolution', methods=['POST'])
def set_camera_resolution():
    data = request.get_json()
    if data.get('index') == '':
        return jsonify(success=False, message="No Cameras")
    index = int(data.get('index'))
    value = data.get('value')

    # Split the resolution string into width and height
    width, height = map(int, value.split('x'))

    success = wrapper.update_config(index, 'camera_resolution_width', width)
    if success:
        success = wrapper.update_config(index, 'camera_resolution_height', height)

    return jsonify(success=success)

@app.route("/calibration")
def calibration():
    global cameras, selected_camera
    return render_template("calibration.html", cameras=cameras, selected_cam=selected_camera)

@app.route('/get_calibration_status')
def get_calibration_status():
    return jsonify(done=wrapper.get_done())

@app.route("/settings")
def settings():
    return render_template("settings.html")

@app.route('/get_mac_settings')
def get_mac_settings():
    return jsonify(wrapper.get_local_settings())

@app.route('/set_mac_settings', methods=['POST'])
def set_mac_settings():
    data = request.get_json()
    if not data:
        return jsonify(success=False), 400
    key = data.get('key')
    value = data.get('value')

    success = wrapper.update_local_settings(key, value)
    return jsonify(success=success)

@app.route('/upload_model', methods=['POST'])
def upload_model():
    file = request.files.get('file')

    if not file:
        return jsonify(success=False), 400

    # Save the uploaded file or process it as needed
    path = f"backend/data/models/{file.filename}"
    file.save(path)
    return jsonify(success=True)

@app.route('/get_models')
def get_models():
    models = os.listdir("backend/data/models")
    selected_model = wrapper.get_selected_model()
    return jsonify(success=True, models=models, selected=selected_model)

@app.route('/upload_tag_layout', methods=['POST'])
def upload_tag_layout():
    file = request.files.get('file')

    if not file:
        return jsonify(success=False), 400

    # Save the uploaded file or process it as needed
    path = f"backend/data/layouts/{file.filename}"
    file.save(path)
    return jsonify(success=True)

@app.route('/get_tag_layouts')
def get_tag_layouts():
    layouts = os.listdir("backend/data/layouts")
    selected_layout = wrapper.get_selected_layout()
    return jsonify(success=True, layouts=layouts, selected=selected_layout)

def start_app():
    # app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5800)
    global server
    server = ServerThread(app, wrapper.get_name())
    server.start()

def stop_app():
    global server
    if server:
        server.stop()
        server.join()
        server = None

@app.route('/restart', methods=['POST'])
def restart():
    stop_app()
    start_app()
    wrapper.restart_nt()
    return jsonify(success=True)

if __name__ == "__main__":
    initialize()
    start_app()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_app()