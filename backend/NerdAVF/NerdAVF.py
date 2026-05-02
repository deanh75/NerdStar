import NerdAVF

import ctypes
from enum import Enum

class NerdAVF:
    class NerdMediaType(Enum):
        audio = "audio"
        auxiliaryPicture = "auxiliaryPicture"
        closedCaption = "closedCaption"
        depthData = "depthData"
        haptic = "haptic"
        metadata = "metadata"
        metadataObject = "metadataObject"
        muxed = "muxed"
        subtitle = "subtitle"
        text = "text"
        timecode = "timecode"
        video = "video"

    class NerdCaptureDevice:
        class DeviceType(Enum):
            builtInWideAngleCamera = "builtInWideAngleCamera"
            continuityCamera = "continuityCamera"
            microphone = "microphone"
            external = "external"
            deskViewCamera = "deskViewCamera"

        class Position(Enum):
            front = "front"
            back = "back"
            unspecified = "unspecified"

        def __init__(self):
            self._uniqueID: str
            self._modelID: str
            self._localizedName: str
            self._manufacturer: str
            self._deviceType: NerdAVF.NerdCaptureDevice.DeviceType
            self._position: NerdAVF.NerdCaptureDevice.Position

        @property
        def uniqueID(self):
            return self._uniqueID
        
        @property
        def modelID(self):
            return self._modelID
        
        @property
        def localizedName(self):
            return self._localizedName 
        
        @property
        def manufacturer(self):
            return self._manufacturer
        
        @property
        def deviceType(self):
            return self._deviceType
        
        @property
        def position(self):
            return self._position
        
        class DiscoverySession:
            def __init__(self, deviceTypes: list[NerdAVF.NerdCaptureDevice.DeviceType], mediaType: NerdAVF.NerdMediaType, position: NerdAVF.NerdCaptureDevice.Position):
                self._devices: list[NerdAVF.NerdCaptureDevice] = []
                
                swift_device_types = [dt.value for dt in deviceTypes]
                swift_media_type = mediaType.value
                swift_position = position.value

                objc.loadBundle(
                    "NerdAVF",
                    bundle_path="./backend/NerdAVF/NerdAVF.framework",
                    module_globals=globals()
                )

                devices = SNerdAVF.discover(swift_device_types, swift_media_type, swift_position)
