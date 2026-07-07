# CMake generated Testfile for 
# Source directory: /home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/cudev/test
# Build directory: /home/nerdstar/NerdStar/backend/opencv/opencv_build/modules/cudev/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_cudev "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_test_cudev" "--gtest_output=xml:opencv_test_cudev.xml")
set_tests_properties(opencv_test_cudev PROPERTIES  LABELS "Extra;opencv_cudev;Accuracy" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/cudev/test/CMakeLists.txt;62;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv_contrib/modules/cudev/test/CMakeLists.txt;0;")
