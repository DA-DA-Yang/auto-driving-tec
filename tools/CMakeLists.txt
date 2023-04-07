#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(tools)

#######################################
#设置Debug或者Release编译
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")
#编译成debug版本，这些都是gcc的编译选项。-O指定优化等级，-g -ggdb表示带上调试信息
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#编译成release版本,-Wall是开启所有编译警告
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
#######################################


#boost
find_package(Boost REQUIRED COMPONENTS thread)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#opencv
find_package(OpenCV)
include_directories(${OpenCV})

#common
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)
aux_source_directory(${COMMON_DIR} COMMON)

#tools/binToPcd
add_executable(BinToPcd ${CMAKE_CURRENT_SOURCE_DIR}/src/binToPcd.cpp ${COMMON})
target_link_libraries(BinToPcd ${PCL_LIBRARIES})

#tools/showBoxes
add_executable(ShowBoxes ${CMAKE_CURRENT_SOURCE_DIR}/src/showBoxes.cpp ${COMMON})
target_link_libraries(ShowBoxes ${PCL_LIBRARIES} ${Boost_THREAD_LIBRARY})

#tools/mapCreate
add_executable(MapCreate ${CMAKE_CURRENT_SOURCE_DIR}/src/mapCreate.cpp)
target_link_libraries(MapCreate ${OpenCV_LIBS})
