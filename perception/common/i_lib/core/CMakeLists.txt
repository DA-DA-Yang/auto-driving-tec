#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(core)

add_library(${PROJECT_NAME} SHARED 
            ${CMAKE_CURRENT_SOURCE_DIR}/i_alloc.h
            ${CMAKE_CURRENT_SOURCE_DIR}/i_basic.h
            ${CMAKE_CURRENT_SOURCE_DIR}/i_blas.h
            ${CMAKE_CURRENT_SOURCE_DIR}/i_constant.h
            ${CMAKE_CURRENT_SOURCE_DIR}/i_rand.h)