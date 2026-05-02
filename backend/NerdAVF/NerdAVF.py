from enum import Enum
import ctypes

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
            builtInWideAngleCamera = "AVCaptureDeviceTypeBuiltInWideAngleCamera"
            continuityCamera = "AVCaptureDeviceTypeContinuityCamera"
            microphone = "AVCaptureDeviceTypeMicrophone"
            external = "AVCaptureDeviceTypeExternal"
            deskViewCamera = "AVCaptureDeviceTypeDeskViewCamera"

        class Position(Enum):
            front = "front"
            back = "back"
            unspecified = "unspecified"

        def __init__(self, uniqueID: str, modelID: str, localizedName: str, manufacturer: str, 
                     deviceType: DeviceType, position: Position):
            self._uniqueID: str = uniqueID
            self._modelID: str = modelID
            self._localizedName: str = localizedName
            self._manufacturer: str = manufacturer
            self._deviceType: NerdAVF.NerdCaptureDevice.DeviceType = deviceType
            self._position: NerdAVF.NerdCaptureDevice.Position = position

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
            def __init__(self, deviceTypes: list["NerdAVF.NerdCaptureDevice.DeviceType"], 
                         mediaType: "NerdAVF.NerdMediaType", position: "NerdAVF.NerdCaptureDevice.Position"):
                self._devices: list[NerdAVF.NerdCaptureDevice] = []
                
                c_device_types = [dt.value for dt in deviceTypes]
                c_media_type = mediaType.value
                c_position = position.value

                api = CNerdAVF("./backend/NerdAVF/libNerdAVF.dylib")
                self._devices = api.discover(c_device_types, c_media_type, c_position)
            
            @property
            def devices(self):
                return self._devices
            
class CNerdDevice(ctypes.Structure):
    _fields_ = [
        ("uniqueID", ctypes.c_char_p),
        ("modelID", ctypes.c_char_p),
        ("localizedName", ctypes.c_char_p),
        ("manufacturer", ctypes.c_char_p),
        ("deviceType", ctypes.c_char_p),
        ("position", ctypes.c_char_p),
    ]

class CNerdAVF:
    def __init__(self, lib_path):
        self.lib = ctypes.CDLL(lib_path)

        self.lib.discover.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int)
        ]

        self.lib.discover.restype = ctypes.POINTER(CNerdDevice)

        # optional free
        try:
            self.lib.nerdavf_free.argtypes = [ctypes.POINTER(CNerdDevice)]
            self._has_free = True
        except:
            self._has_free = False

    def _map_device_type(self, value):
        if value == b"AVCaptureDeviceTypeBuiltInWideAngleCamera":
            return NerdAVF.NerdCaptureDevice.DeviceType.builtInWideAngleCamera
        if value == b"AVCaptureDeviceTypeContinuityCamera":
            return NerdAVF.NerdCaptureDevice.DeviceType.continuityCamera
        if value == b"AVCaptureDeviceTypeMicrophone":
            return NerdAVF.NerdCaptureDevice.DeviceType.microphone
        if value == b"AVCaptureDeviceTypeExternal":
            return NerdAVF.NerdCaptureDevice.DeviceType.external
        if value == b"AVCaptureDeviceTypeDeskViewCamera":
            return NerdAVF.NerdCaptureDevice.DeviceType.deskViewCamera
        return None

    def _map_position(self, value):
        if value == b"front":
            return NerdAVF.NerdCaptureDevice.Position.front
        if value == b"back":
            return NerdAVF.NerdCaptureDevice.Position.back
        return NerdAVF.NerdCaptureDevice.Position.unspecified

    def discover(self, device_types, media_type="video", position="unspecified"):
        count = ctypes.c_int()

        types_array = (ctypes.c_char_p * len(device_types))(
            *[t.encode() for t in device_types]
        )

        devices_ptr = self.lib.discover(
            types_array,
            len(device_types),
            media_type.encode(),
            position.encode(),
            ctypes.byref(count)
        )

        result = []

        for i in range(count.value):
            d = devices_ptr[i]

            result.append(
                NerdAVF.NerdCaptureDevice(
                    uniqueID=d.uniqueID.decode(),
                    modelID=d.modelID.decode(),
                    localizedName=d.localizedName.decode(),
                    manufacturer=d.manufacturer.decode(),
                    deviceType=self._map_device_type(d.deviceType),
                    position=self._map_position(d.position),
                )
            )

        if self._has_free:
            self.lib.nerdavf_free(devices_ptr)

        return result