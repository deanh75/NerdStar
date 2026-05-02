cmake_minimum_required(VERSION 3.10)
project(camera-ctrl)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(openpnp-capture)

add_executable(camera-ctrl main.cpp)
target_link_libraries(camera-ctrl openpnp-capture)