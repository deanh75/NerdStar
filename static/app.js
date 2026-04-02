let currentCamera = "Cam1";

async function loadCameras() {
    const res = await fetch("/api/cameras/");
    const data = await res.json();

    const select = document.getElementById("cameraSelect");
    select.innerHTML = "";

    data.cameras.forEach(cam => {
        const opt = document.createElement("option");
        opt.value = cam.camera_name;
        opt.text = cam.camera_name;
        select.appendChild(opt);
    });

    currentCamera = data.selected;
    select.value = currentCamera;

    loadCameraConfig();
    startStream();
}

function changeCamera() {
    currentCamera = document.getElementById("cameraSelect").value;
    startStream();
    loadCameraConfig();
}

function startStream() {
    const img = document.getElementById("stream");
    img.src = `/api/cameras/stream/${currentCamera}`;
}

async function updateFPS() {
    const res = await fetch(`/api/cameras/fps/${currentCamera}`);
    const data = await res.json();
    document.getElementById("fps").innerText = data.fps.toFixed(1);
}

setInterval(updateFPS, 500);

async function loadCameraConfig() {
    const res = await fetch("/api/cameras/");
    const data = await res.json();

    const cam = data.cameras.find(c => c.camera_name === currentCamera);

    if (!cam) return;

    Object.keys(cam).forEach(key => {
        const el = document.getElementById(key);
        if (el) el.value = cam[key];
    });
}

async function saveConfig() {
    const data = {
        camera_name: document.getElementById("camera_name").value,
        apriltags_enable: document.getElementById("apriltags_enable").value,
        objdetect_enable: document.getElementById("objdetect_enable").value,
        driverCam_enable: document.getElementById("driverCam_enable").value,
        apriltags_stream_port: document.getElementById("apriltags_stream_port").value,
        objdetect_stream_port: document.getElementById("objdetect_stream_port").value
    };

    await fetch("/api/cameras/config", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });
}

window.onload = loadCameras;