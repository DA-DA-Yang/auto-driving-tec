#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(perception)

#设置Debug或者Release编译
set(CMAKE_BUILD_TYPE "Debug")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/point_pillars)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/autoware_pointpillars)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/NCutSegmentation)