cmake_minimum_required(VERSION 3.0)

project(project3)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BUILD_DIRECTORY}/project1")

add_executable(project3 main.cpp quilting.cpp quilting.h)

target_link_libraries(project3 ${OpenCV_LIBS} gflags glog)
