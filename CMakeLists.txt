#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(ait_study)

set(CMAKE_CXX_STANDARD 17)

#######################################
#设置Debug或者Release编译
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
#编译成debug版本，这些都是gcc的编译选项。-O指定优化等级，-g -ggdb表示带上调试信息
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#编译成release版本,-Wall是开启所有编译警告
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#######################################

add_subdirectory(${CMAKE_SOURCE_DIR}/tools)
add_subdirectory(${CMAKE_SOURCE_DIR}/perception)
add_subdirectory(${CMAKE_SOURCE_DIR}/localization)