#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(common)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/i_lib)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/bbox)

