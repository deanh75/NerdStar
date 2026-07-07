# CMake generated Testfile for 
# Source directory: /home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn
# Build directory: /home/nerdstar/NerdStar/backend/opencv/opencv_build/modules/dnn
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_dnn "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_test_dnn" "--gtest_output=xml:opencv_test_dnn.xml")
set_tests_properties(opencv_test_dnn PROPERTIES  LABELS "Main;opencv_dnn;Accuracy" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1364;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;578;ocv_add_accuracy_tests;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_perf_dnn "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml")
set_tests_properties(opencv_perf_dnn PROPERTIES  LABELS "Main;opencv_dnn;Performance" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/performance" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1263;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;598;ocv_add_perf_tests;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_sanity_dnn "/home/nerdstar/NerdStar/backend/opencv/opencv_build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_dnn PROPERTIES  LABELS "Main;opencv_dnn;Sanity" WORKING_DIRECTORY "/home/nerdstar/NerdStar/backend/opencv/opencv_build/test-reports/sanity" _BACKTRACE_TRIPLES "/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVUtils.cmake;1765;add_test;/home/nerdstar/NerdStar/backend/opencv/opencv/cmake/OpenCVModule.cmake;1264;ocv_add_test_from_target;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;598;ocv_add_perf_tests;/home/nerdstar/NerdStar/backend/opencv/opencv/modules/dnn/CMakeLists.txt;0;")
subdirs("../../3rdparty/mlas")
