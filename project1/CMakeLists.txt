cmake_minimum_required(VERSION 3.0)

project(project1)

add_executable(project1 main.cpp)

target_link_libraries(project1 ${OPENCV_LIBS})
target_link_libraries(project1 glog)
