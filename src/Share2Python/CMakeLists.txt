# cmake needs this line
cmake_minimum_required(VERSION 3.1)
 
 
# Define project name
project(opencv_example_project)
 
find_package(OpenCV REQUIRED)
 
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
 
# Declare the executable target built from your sources
 
add_library(python2share SHARED ApiShare_dll.cpp)
target_link_libraries(python2share ${OpenCV_LIBS})