# CMake generated Testfile for 
# Source directory: /home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/shape
# Build directory: /home/nerdstar/NerdStar/backend/opencv/opencv_build/modules/shape
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_shape "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_test_shape" "--gtest_output=xml:opencv_test_shape.xml")
set_tests_properties(opencv_test_shape PROPERTIES  LABELS "Extra;opencv_shape;Accuracy" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1364;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1122;ocv_add_accuracy_tests;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/shape/CMakeLists.txt;2;ocv_define_module;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/shape/CMakeLists.txt;0;")
