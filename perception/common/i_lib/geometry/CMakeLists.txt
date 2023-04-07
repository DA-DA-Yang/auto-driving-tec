#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(geometry)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(${PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/i_line.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/i_plane.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/i_rot.h
                                              ${CMAKE_CURRENT_SOURCE_DIR}/i_util.h)