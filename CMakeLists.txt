cmake_minimum_required(VERSION 3.0)
project(cse555)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

include_directories(/usr/local/include/eigen3 /usr/local/include)
link_directories(/usr/local/lib)

add_subdirectory(project1)
