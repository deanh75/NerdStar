
set(BUILD_SHARED_LIBS "ON")

set(CMAKE_BINARY_DIR "/home/nerdstar/NerdStar/backend/opencv/opencv_build")

set(CMAKE_INSTALL_PREFIX "/usr/local")

set(OpenCV_SOURCE_DIR "/home/nerdstar/NerdStar/backend/opencv/opencv")

set(OPENCV_PC_FILE_NAME "opencv.pc")

set(OPENCV_VERSION_PLAIN "5.1.0")

set(OPENCV_LIB_INSTALL_PATH "lib")

set(OPENCV_INCLUDE_INSTALL_PATH "include/opencv5")

set(OPENCV_3P_LIB_INSTALL_PATH "lib/opencv5/3rdparty")

set(_modules "opencv_stitching;opencv_alphamat;opencv_bgsegm;opencv_bioinspired;opencv_ccalib;opencv_cudabgsegm;opencv_cudaobjdetect;opencv_cudastereo;opencv_dnn_objdetect;opencv_dnn_superres;opencv_dpm;opencv_face;opencv_freetype;opencv_fuzzy;opencv_gapi;opencv_hdf;opencv_hfs;opencv_img_hash;opencv_intensity_transform;opencv_line_descriptor;opencv_quality;opencv_rapid;opencv_reg;opencv_rgbd;opencv_ptcloud;opencv_saliency;opencv_sfm;opencv_signal;opencv_structured_light;opencv_phase_unwrapping;opencv_superres;opencv_cudacodec;opencv_surface_matching;opencv_videostab;opencv_cudaoptflow;opencv_optflow;opencv_cudalegacy;opencv_cudafeatures2d;opencv_calib;opencv_cudawarping;opencv_wechat_qrcode;opencv_objdetect;opencv_xfeatures2d;opencv_shape;opencv_ximgproc;opencv_xobjdetect;opencv_xphoto;opencv_photo;opencv_cudaimgproc;opencv_cudafilters;opencv_cudaarithm;opencv_xstereo;opencv_tracking;opencv_highgui;opencv_datasets;opencv_videoio;opencv_video;opencv_text;opencv_imgcodecs;opencv_features;opencv_dnn;opencv_stereo;opencv_plot;opencv_ml;opencv_imgproc;opencv_geometry;opencv_flann;opencv_core;opencv_cudev")

set(_extra "m;pthread;cudart_static;dl;rt;nppc;nppial;nppicc;nppidei;nppif;nppig;nppim;nppist;nppisu;nppitc;npps;cublas;cudnn;cufft;-L/usr/local/cuda-13.2/lib64;-L/usr/lib/aarch64-linux-gnu")

set(_3rdparty "")

set(TARGET_LOCATION_opencv_stitching "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_stitching.so.5.1.0")

set(TARGET_LOCATION_opencv_alphamat "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_alphamat.so.5.1.0")

set(TARGET_LOCATION_opencv_bgsegm "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_bgsegm.so.5.1.0")

set(TARGET_LOCATION_opencv_bioinspired "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_bioinspired.so.5.1.0")

set(TARGET_LOCATION_opencv_ccalib "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_ccalib.so.5.1.0")

set(TARGET_LOCATION_opencv_cudabgsegm "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudabgsegm.so.5.1.0")

set(TARGET_LOCATION_opencv_cudaobjdetect "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudaobjdetect.so.5.1.0")

set(TARGET_LOCATION_opencv_cudastereo "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudastereo.so.5.1.0")

set(TARGET_LOCATION_opencv_dnn_objdetect "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_dnn_objdetect.so.5.1.0")

set(TARGET_LOCATION_opencv_dnn_superres "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_dnn_superres.so.5.1.0")

set(TARGET_LOCATION_opencv_dpm "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_dpm.so.5.1.0")

set(TARGET_LOCATION_opencv_face "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_face.so.5.1.0")

set(TARGET_LOCATION_opencv_freetype "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_freetype.so.5.1.0")

set(TARGET_LOCATION_opencv_fuzzy "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_fuzzy.so.5.1.0")

set(TARGET_LOCATION_opencv_gapi "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_gapi.so.5.1.0")

set(TARGET_LOCATION_opencv_hdf "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_hdf.so.5.1.0")

set(TARGET_LOCATION_opencv_hfs "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_hfs.so.5.1.0")

set(TARGET_LOCATION_opencv_img_hash "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_img_hash.so.5.1.0")

set(TARGET_LOCATION_opencv_intensity_transform "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_intensity_transform.so.5.1.0")

set(TARGET_LOCATION_opencv_line_descriptor "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_line_descriptor.so.5.1.0")

set(TARGET_LOCATION_opencv_quality "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_quality.so.5.1.0")

set(TARGET_LOCATION_opencv_rapid "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_rapid.so.5.1.0")

set(TARGET_LOCATION_opencv_reg "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_reg.so.5.1.0")

set(TARGET_LOCATION_opencv_rgbd "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_rgbd.so.5.1.0")

set(TARGET_LOCATION_opencv_ptcloud "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_ptcloud.so.5.1.0")

set(TARGET_LOCATION_opencv_saliency "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_saliency.so.5.1.0")

set(TARGET_LOCATION_opencv_sfm "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_sfm.so.5.1.0")

set(TARGET_LOCATION_opencv_signal "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_signal.so.5.1.0")

set(TARGET_LOCATION_opencv_structured_light "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_structured_light.so.5.1.0")

set(TARGET_LOCATION_opencv_phase_unwrapping "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_phase_unwrapping.so.5.1.0")

set(TARGET_LOCATION_opencv_superres "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_superres.so.5.1.0")

set(TARGET_LOCATION_opencv_cudacodec "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudacodec.so.5.1.0")

set(TARGET_LOCATION_opencv_surface_matching "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_surface_matching.so.5.1.0")

set(TARGET_LOCATION_opencv_videostab "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_videostab.so.5.1.0")

set(TARGET_LOCATION_opencv_cudaoptflow "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudaoptflow.so.5.1.0")

set(TARGET_LOCATION_opencv_optflow "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_optflow.so.5.1.0")

set(TARGET_LOCATION_opencv_cudalegacy "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudalegacy.so.5.1.0")

set(TARGET_LOCATION_opencv_cudafeatures2d "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudafeatures2d.so.5.1.0")

set(TARGET_LOCATION_opencv_calib "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_calib.so.5.1.0")

set(TARGET_LOCATION_opencv_cudawarping "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudawarping.so.5.1.0")

set(TARGET_LOCATION_opencv_wechat_qrcode "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_wechat_qrcode.so.5.1.0")

set(TARGET_LOCATION_opencv_objdetect "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_objdetect.so.5.1.0")

set(TARGET_LOCATION_opencv_xfeatures2d "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_xfeatures2d.so.5.1.0")

set(TARGET_LOCATION_opencv_shape "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_shape.so.5.1.0")

set(TARGET_LOCATION_opencv_ximgproc "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_ximgproc.so.5.1.0")

set(TARGET_LOCATION_opencv_xobjdetect "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_xobjdetect.so.5.1.0")

set(TARGET_LOCATION_opencv_xphoto "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_xphoto.so.5.1.0")

set(TARGET_LOCATION_opencv_photo "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_photo.so.5.1.0")

set(TARGET_LOCATION_opencv_cudaimgproc "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudaimgproc.so.5.1.0")

set(TARGET_LOCATION_opencv_cudafilters "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudafilters.so.5.1.0")

set(TARGET_LOCATION_opencv_cudaarithm "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudaarithm.so.5.1.0")

set(TARGET_LOCATION_opencv_xstereo "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_xstereo.so.5.1.0")

set(TARGET_LOCATION_opencv_tracking "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_tracking.so.5.1.0")

set(TARGET_LOCATION_opencv_highgui "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_highgui.so.5.1.0")

set(TARGET_LOCATION_opencv_datasets "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_datasets.so.5.1.0")

set(TARGET_LOCATION_opencv_videoio "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_videoio.so.5.1.0")

set(TARGET_LOCATION_opencv_video "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_video.so.5.1.0")

set(TARGET_LOCATION_opencv_text "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_text.so.5.1.0")

set(TARGET_LOCATION_opencv_imgcodecs "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_imgcodecs.so.5.1.0")

set(TARGET_LOCATION_opencv_features "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_features.so.5.1.0")

set(TARGET_LOCATION_opencv_dnn "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_dnn.so.5.1.0")

set(TARGET_LOCATION_opencv_stereo "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_stereo.so.5.1.0")

set(TARGET_LOCATION_opencv_plot "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_plot.so.5.1.0")

set(TARGET_LOCATION_opencv_ml "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_ml.so.5.1.0")

set(TARGET_LOCATION_opencv_imgproc "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_imgproc.so.5.1.0")

set(TARGET_LOCATION_opencv_geometry "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_geometry.so.5.1.0")

set(TARGET_LOCATION_opencv_flann "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_flann.so.5.1.0")

set(TARGET_LOCATION_opencv_core "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_core.so.5.1.0")

set(TARGET_LOCATION_opencv_cudev "/home/nerdstar/NerdStar/backend/opencv/opencv_build/lib/libopencv_cudev.so.5.1.0")
