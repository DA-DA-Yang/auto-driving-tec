#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(bbox)

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(${PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/bounding_box.cpp)
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES})

