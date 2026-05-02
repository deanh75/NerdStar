#import <AVFoundation/AVFoundation.h>
#include "nerdavf.hpp"

static const char* copyNSString(NSString* str) {
    return strdup([str UTF8String]);
}

extern "C" {

CNerdDevice* discover(
    const char** deviceTypes,
    int deviceTypeCount,
    const char* mediaType,
    const char* position,
    int* outCount
) {
    NSMutableArray<AVCaptureDeviceType>* types = [NSMutableArray new];

    for (int i = 0; i < deviceTypeCount; i++) {
        NSString* t = [NSString stringWithUTF8String:deviceTypes[i]];
        [types addObject:(AVCaptureDeviceType)t];
    }

    AVCaptureDevicePosition pos = AVCaptureDevicePositionUnspecified;
    if (strcmp(position, "front") == 0) pos = AVCaptureDevicePositionFront;
    if (strcmp(position, "back") == 0) pos = AVCaptureDevicePositionBack;

    AVCaptureDeviceDiscoverySession* session =
        [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:types
            mediaType:(AVMediaType)@(mediaType)
            position:pos];

    NSArray* devices = session.devices;

    *outCount = (int)devices.count;

    CNerdDevice* result = (CNerdDevice*)malloc(sizeof(CNerdDevice) * devices.count);

    for (int i = 0; i < devices.count; i++) {
        AVCaptureDevice* d = devices[i];
        
        NSString* pos = @"unspecified";
        if (d.position == AVCaptureDevicePositionFront) pos = @"front";
        if (d.position == AVCaptureDevicePositionBack) pos = @"back";

        result[i].uniqueID = copyNSString(d.uniqueID);
        result[i].modelID = copyNSString(d.modelID);
        result[i].localizedName = copyNSString(d.localizedName);
        result[i].manufacturer = copyNSString(d.manufacturer);
        result[i].deviceType = copyNSString([d.deviceType description]);
        result[i].position = copyNSString(pos);
    }

    return result;
}

void nerdavf_free(CNerdDevice* devices) {
    free(devices);
}

AVCaptureDevice* nerdToAVF(CNerdDevice dev) {
    NSMutableArray<AVCaptureDeviceType>* type = [NSMutableArray new];
    [type addObject:[NSString stringWithUTF8String:dev.deviceType]];
    
    AVCaptureDevicePosition pos = AVCaptureDevicePositionUnspecified;
    if (strcmp(dev.position, "front") == 0) pos = AVCaptureDevicePositionFront;
    if (strcmp(dev.position, "back") == 0) pos = AVCaptureDevicePositionBack;
    
    AVCaptureDeviceDiscoverySession* session =
        [AVCaptureDeviceDiscoverySession
            discoverySessionWithDeviceTypes:type
            mediaType:AVMediaTypeVideo
            position:pos];
    
    for (AVCaptureDevice *device in session.devices) {
        if ([device.uniqueID isEqualToString:[NSString stringWithUTF8String:dev.uniqueID]]) {
            return device;
        }
    }
    return nil;
}

bool lockForConfig(CNerdDevice nerdDevice) {
    AVCaptureDevice *avf = nerdToAVF(nerdDevice);
    if (!avf) {
        return false;
    }
    return [avf lockForConfiguration:nil];
}

bool unlockForConfig(CNerdDevice nerdDevice) {
    AVCaptureDevice *avf = nerdToAVF(nerdDevice);
    if (!avf) {
        return false;
    }
    [avf unlockForConfiguration];
    return true;
}
}
