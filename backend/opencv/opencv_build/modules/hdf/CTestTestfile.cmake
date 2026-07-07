# CMake generated Testfile for 
# Source directory: /home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/hdf
# Build directory: /home/nerdstar/NerdStar/backend/opencv/opencv_build/modules/hdf
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_hdf "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_test_hdf" "--gtest_output=xml:opencv_test_hdf.xml")
set_tests_properties(opencv_test_hdf PROPERTIES  LABELS "Extra;opencv_hdf;Accuracy" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1364;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1122;ocv_add_accuracy_tests;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/hdf/CMakeLists.txt;33;ocv_define_module;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/hdf/CMakeLists.txt;0;")
