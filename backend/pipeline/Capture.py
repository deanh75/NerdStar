# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file at
# the root directory of this project.

from dataclasses import replace
import gc
from glob import glob
import threading
import time
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import weakref
import cv2
import numpy as np
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib 
import pyds

from backend.config.config import CameraConfig, ConfigStore

class Capture:
    """Interface for receiving camera frames."""

    def __init__(self, method="by-path") -> None:
        raise NotImplementedError
    
    def get_frame(self, cam_id: str, configs: List[ConfigStore]) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
        raise NotImplementedError
    
    def getCameras() -> list: 
        """Return a list of available cameras."""
        raise NotImplementedError
    
    def stop(self) -> None: 
        raise NotImplementedError

    @classmethod
    def _config_changed(cls, config_a: ConfigStore, config_b: ConfigStore) -> bool:
        if config_a == None and config_b == None:
            return False
        if config_a == None or config_b == None:
            return True

        camera_a = config_a.camera_config
        camera_b = config_b.camera_config

        return (
            camera_a.camera_auto_white_balance != camera_b.camera_auto_white_balance
            or camera_a.camera_white_balance != camera_b.camera_white_balance
            or camera_a.camera_auto_exposure != camera_b.camera_auto_exposure
            or camera_a.camera_exposure != camera_b.camera_exposure
            or camera_a.camera_gain != camera_b.camera_gain
        )
    
    @classmethod
    def _res_changed(cls, config_a: CameraConfig, config_b: CameraConfig) -> bool:
        if config_a == None and config_b == None:
            return False
        if config_a == None or config_b == None:
            return True
        
        return (
            config_a.camera_resolution_height != config_b.camera_resolution_height
            or config_a.camera_resolution_width != config_b.camera_resolution_width
            or config_a.camera_max_fps != config_b.camera_max_fps
            or config_a.camera_id != config_b.camera_id
        )

class JetsonCapture(Capture):
    def __init__(self, method="by-path") -> None:
        self._last_configs: Dict[str, ConfigStore] = {}
        self._pipelines: Dict[str, Optional[dict]] = {}
        self._base_dir = f"/dev/v4l/{method}"
        self._pgie_config_path = None
        pass

    def getCameras(self) -> list: 
        if not os.path.isdir(self._base_dir):
            return []
        
        candidates = sorted(glob(os.path.join(self._base_dir, "*-video-index0")))
 
        seen_targets = set()
        result = []
        for path in candidates:
            target = os.path.realpath(path)
            if target in seen_targets:
                continue
            seen_targets.add(target)
            name = Path(path).name
            result.append(name)
        return result

        
    def stop(self) -> None:
        try:
            if self._pipelines != None:
                for pipeline in self._pipelines.values():
                    pipeline.set_state(Gst.State.NULL)
                self._pipelines.clear()
                self._last_configs.clear()
                gc.collect()
        except Exception as e:
            print("Stop error:", e)

    def build_ctrls(self, cam_cfg: CameraConfig) -> str:
        controls = [
            f"auto_exposure={1 if not cam_cfg.camera_auto_exposure else 3}"
        ]
        if not cam_cfg.camera_auto_exposure and cam_cfg.camera_exposure is not None:
            controls.append(f"exposure_time_absolute={cam_cfg.camera_exposure}")
        if cam_cfg.camera_gain is not None:
            controls.append(f"gain={cam_cfg.camera_gain}")
        controls.append(
            f"white_balance_automatic={1 if cam_cfg.camera_auto_white_balance else 0}"
        )
        if not cam_cfg.camera_auto_white_balance and cam_cfg.camera_white_balance is not None:
            controls.append(f"white_balance_temperature={cam_cfg.camera_white_balance}")
        return "c," + ",".join(controls)

    def change_setting(self, cam_id: str, setting: str, value) -> None:
        state = self._get_state(cam_id)
        if state is None:
            print(f"Camera {cam_id} not found")
            return
        
        if setting == "camera_auto_exposure":
            setting = "auto_exposure"
            value = 1 if not value else 3
        elif setting == "camera_exposure":
            setting = "exposure_time_absolute"
        elif setting == "camera_gain":
            setting = "gain"
        elif setting == "camera_auto_white_balance":
            setting = "white_balance_automatic"
            value = 1 if value else 0
        elif setting == "camera_white_balance":
            setting = "white_balance_temperature"
        else:
            print(f"Unknown setting {setting}")
            return

        ctrl = "c," + f"{setting}={value}"

        iterator = state["pipeline"].iterate_elements()
        while True:
            success, el = iterator.next()
            if not success:
                break

            factory = el.get_factory()
            if factory is not None and factory.get_name() == "v4l2src":
                el.set_property("extra-controls", Gst.Structure.new_from_string(ctrl))
                break

    def get_frame(self, config: ConfigStore) -> None:
        cam_id = config.camera_config.camera_id if config else ""
        last_config = self._last_configs.get(cam_id)
        state = self._pipelines.get(cam_id)
    
        if config is None:
            print(f"No config found for camera {cam_id}")
            return

        if state is not None and self._config_changed(last_config, config):
            diff = [k for k, v in config.camera_config.__dict__.items() if v != last_config.camera_config.__dict__.get(k)]
            for setting in diff:
                value = getattr(config.camera_config, setting)
                self.change_setting(cam_id, setting, value)
                last_config.camera_config.__dict__[setting] = value

        # -- restart if config changed --
        if state is not None and self._res_changed(last_config.camera_config, config.camera_config):
            print("Restarting capture session")
            state["pipeline"].set_state(Gst.State.NULL)
            state["pipeline"].get_state(Gst.CLOCK_TIME_NONE)
            state["loop"].quit()
            state = None
            self._pipelines[cam_id] = None
            last_config = ConfigStore(
                replace(config.camera_config),
                replace(config.remote_config),
                config.camera_config_source,
                config.remote_config_source,
            )
            self._last_configs[cam_id] = last_config
            time.sleep(2.5)
    
        # -- restart if the pipeline thread died (GStreamer bus ERROR/EOS) --
        if state is not None and not state["thread"].is_alive():
            print("Capture session died, restarting")
            state["pipeline"].set_state(Gst.State.NULL)
            state = None
            self._pipelines[cam_id] = None
            time.sleep(7.5)
    
        # -- start a fresh pipeline if none is running --
        if state is None:
            if config.camera_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                device = None
                for dev in self.getCameras():
                    if dev == config.camera_config.camera_id:
                        device = dev
                        break
    
                if device is None:
                    print(f"Camera {config.camera_config.camera_id} not found among connected devices")
                else:
                    try:
                        cam_cfg = config.camera_config
                        width = cam_cfg.camera_resolution_width
                        height = cam_cfg.camera_resolution_height
                        fps = cam_cfg.camera_max_fps
    
                        extra_controls = self.build_ctrls(cam_cfg)
    
                        Gst.init(None)
                        pipeline = Gst.Pipeline()
    
                        def make(factory, name):
                            el = Gst.ElementFactory.make(factory, name)
                            if not el:
                                raise RuntimeError(f"Failed to create element {factory} ({name})")
                            pipeline.add(el)
                            return el
    
                        self.src = make("v4l2src", f"usb-camera-{config.camera_config.camera_id}")
                        self.src.set_property("device", f"{self._base_dir}/{device}")
                        self.src.set_property("extra-controls", Gst.Structure.new_from_string(extra_controls))
                        self.src.set_property("io-mode", 2) # mmap mode
    
                        src_caps = make("capsfilter", "src-caps")
                        # image/jpeg routes this to hardware MJPEG decode via nvv4l2decoder below
                        src_caps.set_property(
                            "caps",
                            Gst.Caps.from_string(f"image/jpeg,width={width},height={height},framerate={fps}/1"),
                        )
    
                        jpegparse = make("jpegparse", "jpegparse")
                        decoder = make("nvv4l2decoder", "decoder")
                        decoder.set_property("mjpeg", 1)
    
                        convert1 = make("nvvideoconvert", "convert-to-rgba")
                        convert1_caps = make("capsfilter", "convert1-caps")
                        convert1_caps.set_property(
                            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA")
                        )
    
                        streammux = make("nvstreammux", "streammux")
                        streammux.set_property("width", width)
                        streammux.set_property("height", height)
                        streammux.set_property("batch-size", 1)
                        streammux.set_property("batched-push-timeout", 8400)
                        streammux.set_property("live-source", True)
    
                        #TODO: What is pgie????
                        pgie = None
                        if self._pgie_config_path:
                            pgie = make("nvinfer", "pgie")
                            pgie.set_property("config-file-path", self._pgie_config_path)

                        tee = make("tee", "tee")
    
                        # Internal GStreamer buffering, capped at exactly one buffer,
                        # dropping old ones -- this is what enforces "only buffer one
                        # frame" at the pipeline level.
                        queue_probe = make("queue", "queue-probe")
                        queue_probe.set_property("leaky", 1)
                        queue_probe.set_property("max-size-buffers", 1)
                        queue_probe.set_property("max-size-bytes", 0)
                        queue_probe.set_property("max-size-time", 0)
                        sink = make("appsink", "probe-sink")
                        sink.set_property("emit-signals", True)
                        sink.set_property("max-buffers", 1)
                        sink.set_property("drop", True)
                        sink.set_property("sync", False)
    
                        # -- link main chain --
                        self.src.link(src_caps)
                        src_caps.link(jpegparse)
                        jpegparse.link(decoder)
                        decoder.link(convert1)
                        convert1.link(convert1_caps)
    
                        sinkpad = streammux.get_request_pad("sink_0")
                        srcpad = convert1_caps.get_static_pad("src")
                        srcpad.link(sinkpad)
    
                        if pgie:
                            streammux.link(pgie)
                            pgie.link(tee)
                        else:
                            streammux.link(tee)
    
                        tee_pad_probe = tee.get_request_pad("src_%u")
                        tee_pad_probe.link(queue_probe.get_static_pad("sink"))
                        queue_probe.link(sink)
    
                        lock = threading.Lock()
                        new_state = {
                            "pipeline": pipeline, "loop": None, "thread": None,
                            "lock": lock, "mat": None,
                        }
    
                        def probe_callback(pad, info, u_data):
                            gst_buffer = info.get_buffer()
                            if not gst_buffer:
                                print("Unable to get GstBuffer")
                                return Gst.PadProbeReturn.OK
    
                            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
                            if not batch_meta:
                                print("Unable to get batch meta")
                                return Gst.PadProbeReturn.OK
                            l_frame = batch_meta.frame_meta_list
                            while l_frame is not None:
                                try:
                                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                                except StopIteration:
                                    continue

                                try:
                                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                                    rgba = np.array(n_frame, copy=True, order="C")

                                    if new_state["mat"] is None:
                                        new_state["mat"] = rgba  # first frame: allocate
                                    else:
                                        new_state["mat"][:] = rgba
                                except Exception as e:
                                    print(f"Error occurred while processing frame: {e}")
    
                                try:
                                    l_frame = l_frame.next
                                except StopIteration:
                                    break
                            return Gst.PadProbeReturn.OK
    
                        probe_pad = (pgie if pgie else streammux).get_static_pad("src")
                        probe_pad.add_probe(Gst.PadProbeType.BUFFER, probe_callback, None)
    
                        loop = GLib.MainLoop()
                        bus = pipeline.get_bus()
                        bus.add_signal_watch()
    
                        def on_message(bus, message):
                            t = message.type
                            if t == Gst.MessageType.EOS:
                                loop.quit()
                            elif t == Gst.MessageType.ERROR:
                                err, debug = message.parse_error()
                                print(f"GStreamer error: {err}, {debug}")
                                loop.quit()
    
                        bus.connect("message", on_message)

                        pipeline.set_state(Gst.State.PLAYING)
                        thread = threading.Thread(target=loop.run, daemon=True)
                        thread.start()
    
                        new_state["loop"] = loop
                        new_state["thread"] = thread
    
                        state = new_state
                        self._pipelines[cam_id] = state
                    except RuntimeError as e:
                        print(f"Failed to open pipeline for camera {cam_id}: {e}")
                        state = None
                        self._pipelines[cam_id] = None
    
        if last_config is None:
            last_config = ConfigStore(
                replace(config.camera_config),
                replace(config.remote_config),
                config.camera_config_source,
                config.remote_config_source,
            )
            self._last_configs[cam_id] = last_config

        if state is not None and self._config_changed(last_config, config):
            ctrls = self.build_ctrls(config.camera_config)
            self.src.set_property("extra-controls", Gst.Structure.new_from_string(ctrls))
            last_config = ConfigStore(
                replace(config.camera_config),
                replace(config.remote_config),
                config.camera_config_source,
                config.remote_config_source,
            )
            self._last_configs[cam_id] = last_config
            
    
    def get_cpu(self, cam_id: str) -> Optional[cv2.Mat]:
        state = self._get_state(cam_id)
        if state is None:
            return None
        mat = state["mat"]
        if mat is None:
            return None
        return cv2.cvtColor(mat, cv2.COLOR_RGBA2BGR)
    
    def get_gpu(self, cam_id: str) -> Optional[cv2.cuda.GpuMat]:
        state = self._get_state(cam_id)
        if state is None or state["mat"] is None:
            return None
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            raise RuntimeError("OpenCV has no CUDA-enabled devices -- check cv2.getBuildInformation()")
        mat = state["mat"]
        if not state.get("_page_locked"):
            cv2.cuda.registerPageLocked(mat)
            state["_page_locked"] = True
            state["_page_locked_mat"] = mat

            weakref.finalize(mat, cv2.cuda.unregisterPageLocked, mat)

        gpu_mat = cv2.cuda.GpuMat()
        gpu_mat.upload(mat)
        return cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_RGBA2RGB)
    
    def _get_state(self, cam_id: str) -> Optional[dict]:
        pipe = self._pipelines.get(cam_id)
        if pipe is None:
            return None
        return pipe
