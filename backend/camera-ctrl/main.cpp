#include <iostream>

#include "openpnp-capture/common/logging.h"
#include "openpnp-capture/include/openpnp-capture.h"
#include "openpnp-capture/mac/uvcctrl.h"

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::cerr << "Usage: camera_ctrl <vid> <pid> <location> <auto wb> <wb> <auto exposure> <exposure> <gain>" <<
                std::endl;
        std::cerr << "\tvid: IOKit Vendor ID in hex" << std::endl;
        std::cerr << "\tpid: IOKit Product ID in hex" << std::endl;
        std::cerr << "\tlocation: IOKit Location ID in hex, or 0 to select the first found location" << std::endl;
        std::cerr << "\tauto wb: 1 to enable auto white balance, 0 to disable" << std::endl;
        std::cerr << "\twb: white balance value, decimal" << std::endl;
        std::cerr << "\tauto exposure: 1 to enable auto exposure, 0 to disable" << std::endl;
        std::cerr << "\texposure: exposure value, decimal" << std::endl;
        std::cerr << "\tgain: gain value, decimal" << std::endl;
        return 1;
    }

    unsigned int vid = std::stoul(argv[1], nullptr, 16);
    unsigned int pid = std::stoul(argv[2], nullptr, 16);
    unsigned int location = std::stoul(argv[3], nullptr, 16);
    int autoWB = std::stoi(argv[4], nullptr, 10);
    int wb = std::stoi(argv[5], nullptr, 10);
    int autoExposure = std::stoi(argv[6], nullptr, 10);
    int exposure = std::stoi(argv[7], nullptr, 10);
    int gain = std::stoi(argv[8], nullptr, 10);

    setLogLevel(LOG_VERBOSE);
    std::shared_ptr<UVCCtrl> ctrl(UVCCtrl::create(vid, pid, location));
    if (!ctrl) {
        std::cerr << "Failed to create UVC controller for device with VID: " << std::hex << vid
                  << ", PID: " << std::hex << pid
                  << ", Location: " << std::hex << location
                  << std::dec << std::endl;
        return 1;
    }

    // Read out initial settings
    bool oldAutoWB;
    int oldWB;
    bool oldAutoExposure;
    int oldExposure;
    int oldGain;

    ctrl->getAutoProperty(CAPPROPID_WHITEBALANCE, &oldAutoWB);
    ctrl->getProperty(CAPPROPID_WHITEBALANCE, &oldWB);
    ctrl->getAutoProperty(CAPPROPID_EXPOSURE, &oldAutoExposure);
    ctrl->getProperty(CAPPROPID_EXPOSURE, &oldExposure);
    ctrl->getProperty(CAPPROPID_GAIN, &oldGain);

    std::cout << "Auto white balance was " << static_cast<int>(oldAutoWB) << ", will set to " <<
            autoWB << std::endl;
    std::cout << "White balance was " << oldWB << ", will set to " << wb << std::endl;
    std::cout << "Auto exposure was " << static_cast<int>(oldAutoExposure) << ", will set to " <<
            autoExposure << std::endl;
    std::cout << "Exposure was " << oldExposure << ", will set to " << exposure << std::endl;
    std::cout << "Gain was " << oldGain << ", will set to " << gain << std::endl;

    // Set new settings
    ctrl->setAutoProperty(CAPPROPID_WHITEBALANCE, autoWB);
    ctrl->setProperty(CAPPROPID_WHITEBALANCE, wb);
    ctrl->setAutoProperty(CAPPROPID_EXPOSURE, autoExposure);
    ctrl->setProperty(CAPPROPID_EXPOSURE, exposure);
    ctrl->setProperty(CAPPROPID_GAIN, gain);

    std::cout << "Done" << std::endl;
    return 0;
}