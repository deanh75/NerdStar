#include <iostream>

// Match your struct definition
typedef struct CNerdDevice {
    const char* uniqueID;
    const char* modelID;
    const char* localizedName;
    const char* manufacturer;
    const char* deviceType;
    const char* position;
} CNerdDevice;

// Your Objective-C++ function
extern "C" CNerdDevice* discover(
    const char** deviceTypes,
    int deviceTypeCount,
    const char* mediaType,
    const char* position,
    int* outCount
);

int main() {
    int count = 0;

    const char* deviceTypes[] = {
        "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        "AVCaptureDeviceTypeExternal"
    };

    const char* mediaType = "video";
    const char* position = "front";

    CNerdDevice* devices = discover(
        deviceTypes,
        2,
        mediaType,
        position,
        &count
    );

    std::cout << "Found devices: " << count << std::endl;

    for (int i = 0; i < count; i++) {
        CNerdDevice& d = devices[i];

        std::cout << "---- Device " << i << " ----\n";
        std::cout << "Unique ID: " << (d.uniqueID ? d.uniqueID : "null") << "\n";
        std::cout << "Model ID: " << (d.modelID ? d.modelID : "null") << "\n";
        std::cout << "Name: " << (d.localizedName ? d.localizedName : "null") << "\n";
        std::cout << "Manufacturer: " << (d.manufacturer ? d.manufacturer : "null") << "\n";
        std::cout << "Type: " << (d.deviceType ? d.deviceType : "null") << "\n";
        std::cout << "Position: " << (d.position ? d.position : "null") << "\n";
    }

    // If your Objective-C++ allocates with malloc, free it here
    free(devices);

    return 0;
}
