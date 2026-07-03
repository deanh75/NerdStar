#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* uniqueID;
    const char* modelID;
    const char* localizedName;
    const char* manufacturer;
    const char* deviceType;
    const char* position;
} CNerdDevice;

// returns array of devices (allocated internally)
CNerdDevice* discover(
    const char** deviceTypes,
    int deviceTypeCount,
    const char* mediaType,
    const char* position,
    int* outCount
);

void nerdavf_free(CNerdDevice* devices);

bool lockForConfig(CNerdDevice nerdDevice);
bool unlockForConfig(CNerdDevice nerdDevice);

#ifdef __cplusplus
}
#endif
