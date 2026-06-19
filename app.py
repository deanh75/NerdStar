# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

import asyncio
import subprocess
import atexit
import os
import threading
import time

from fastapi import FastAPI, Query, Request, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription
from backend.CameraTrack import CameraTrack
from backend.pipeline.RobotPoseEstimator import RobotPoseEstimator
from backend.wrapper import Wrapper

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
pcs = set()

wrapper = Wrapper()
atexit.register(wrapper._capture.stop)

selected_camera = None
camera_supplier = lambda: selected_camera
init = False

state = {"index": -1}

def initialize():
    global init, cameras, selected_camera
    if not init:
        cameras = wrapper.get_cameras()
        selected_camera = cameras[0] if cameras else None
        estimator = RobotPoseEstimator(wrapper.local_config)
        
        for i in range(len(cameras)):
            threading.Thread(target=wrapper.start_backend, args=(i,), daemon=True).start()

        threading.Thread(target=wrapper.estimate, args=(estimator,), daemon=True).start()

        init = True

@app.get("/")
def camera(request: Request):
    global cameras, selected_camera
    return templates.TemplateResponse(request, "camera.html", {
        "cameras": cameras, 
        "selected_cam": selected_camera
    })

@app.post("/set_camera")
def set_camera(camera: str = Form(...)):
    global selected_camera
    if camera in cameras:
        selected_camera = camera
    return JSONResponse({"success": True})

@app.post('/offer')
async def offer(request: Request):
    params = await request.json()

    offer = RTCSessionDescription(
        sdp=params["sdp"], 
        type=params["type"]
    )

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    track = CameraTrack(wrapper, camera_supplier)
    pc.addTrack(track)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

@app.post('/rename_camera')
async def rename_camera_route(request: Request):
    data = await request.json()
    index = int(data.get('index'))
    new_name: str = data.get('name', '').strip()

    if new_name and wrapper.update_config(index, 'camera_name', new_name):
        global cameras, selected_camera
        selected_camera = new_name  # Update selected camera to the renamed one
        cameras[index] = new_name
        return JSONResponse({"success": True})
    return JSONResponse({"success": False}, status_code=400)

@app.get('/get_camera_settings')
def get_camera_settings(index: int = Query(...)):
    print(f"Getting settings for camera: {index}")
    return wrapper.get_camera_settings(index)

@app.get('/get_cviz_settings')
def get_cviz_settings(index: int = Query(...)):
    print(f"Getting cam-viz settings for camera: {index}")
    return wrapper.get_cviz_settings(index)

@app.post('/set_camera_setting')
async def set_camera_setting(request: Request):
    data = await request.json()
    if data.get('index') == '':
        return JSONResponse({"success": False, "message": "No Cameras"}, status_code=400)
    index = int(data.get('index'))
    key = data.get('key')
    value = data.get('value')

    success = wrapper.update_config(index, key, value)
    return JSONResponse({"success": success})

@app.websocket("/ws/apriltag_data")
async def apriltag_data(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = wrapper.get_apriltag_data(camera_supplier)
            await ws.send_json(data)
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        pass

@app.websocket("/ws/obj_data")
async def obj_data(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = wrapper.get_obj_data(camera_supplier)
            await ws.send_json(data)
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        pass

@app.post('/set_camera_resolution')
async def set_camera_resolution(request: Request):
    data = await request.json()
    if data.get('index') == '':
        return JSONResponse({"success": False, "message": "No Cameras"}, status_code=400)
    index = int(data.get('index'))
    value = data.get('value')

    # Split the resolution string into width and height
    width, height = map(int, value.split('x'))

    success = wrapper.update_config(index, 'camera_resolution_width', width)
    if success:
        success = wrapper.update_config(index, 'camera_resolution_height', height)

    return JSONResponse({"success": success})

@app.get("/calibration")
def calibration(request: Request):
    global cameras, selected_camera
    return templates.TemplateResponse(request, "calibration.html", {
        "cameras": cameras,
        "selected_cam": selected_camera
    })

@app.get('/get_calibration_status')
def get_calibration_status():
    return JSONResponse({"done": wrapper.get_done()})

@app.get("/settings")
def settings(request: Request):
    return templates.TemplateResponse(request, "settings.html", {})

@app.get('/get_local_settings')
def get_local_settings():
    return JSONResponse(wrapper.get_local_settings())

@app.post('/set_local_settings')
async def set_local_settings(request: Request):
    data = await request.json()
    if not data:
        return JSONResponse({"success": False}, status_code=400)
    key = data.get('key')
    value = data.get('value')

    success = wrapper.update_local_settings(key, value)
    return JSONResponse({"success": success})

@app.post('/upload_model')
async def upload_model(file: UploadFile = File(...)):
    path = f"backend/data/models/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    return JSONResponse({"success": True})

@app.get('/get_models')
def get_models():
    models = os.listdir("backend/data/models")
    selected_model = wrapper.get_selected_model()
    return JSONResponse({"success": True, "models": models, "selected": selected_model})

@app.post('/upload_tag_layout')
async def upload_tag_layout(file: UploadFile = File(...)):
    path = f"backend/data/layouts/{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    return JSONResponse({"success": True})

@app.get('/get_tag_layouts')
def get_tag_layouts():
    layouts = os.listdir("backend/data/layouts")
    selected_layout = wrapper.get_selected_layout()
    return JSONResponse({"success": True, "layouts": layouts, "selected": selected_layout})

def set_hostname(hostname: str):
    print(f"Server started: {hostname}")
    subprocess.run(["sudo", "scutil", "--set", "HostName", hostname], check=True)
    subprocess.run(["sudo", "scutil", "--set", "LocalHostName", hostname], check=True)
    subprocess.run(["sudo", "scutil", "--set", "ComputerName", hostname], check=True)

def start_app():
    # app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5800)
    global server  
    config = uvicorn.Config(app=app, host='0.0.0.0', port=5800, log_level="info")
    server = uvicorn.Server(config)
    set_hostname(wrapper.get_name())
    
    threading.Thread(target=server.run, daemon=True).start()

def stop_app():
    global server
    if server:
        server.should_exit = True
        server = None

@app.post('/restart')
def restart():
    stop_app()
    start_app()
    wrapper.restart_nt()
    return JSONResponse({"success": True})

if __name__ == "__main__":
    initialize()
    start_app()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_app()