cmake_minimum_required(VERSION 3.0)

project(project1)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${BUILD_DIRECTORY}/project1")

add_executable(project1 main.cpp)

message(${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

target_link_libraries(project1 ${OpenCV_LIBS})
target_link_libraries(project1 glog)
target_link_libraries(project1 gflags)
