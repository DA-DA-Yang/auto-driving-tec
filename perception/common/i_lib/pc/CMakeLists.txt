#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(pc)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(${PROJECT_NAME} SHARED ${CMAKE_CURRENT_SOURCE_DIR}/i_ground.cc
                                   ${CMAKE_CURRENT_SOURCE_DIR}/i_struct_s.h
                                   ${CMAKE_CURRENT_SOURCE_DIR}/i_util.h)