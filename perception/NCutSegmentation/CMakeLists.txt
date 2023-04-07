#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(NCutSegmentation)

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
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)
aux_source_directory(${COMMON_DIR} COMMON)

#src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
aux_source_directory(${SRC_DIR} SRC)

#main
add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${SRC} ${COMMON})
target_include_directories(main PRIVATE SRC_DIR COMMON_DIR)
target_link_libraries(main i_lib bbox ${PCL_LIBRARIES} ${Boost_THREAD_LIBRARY} ${OpenCV_LIBS})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

