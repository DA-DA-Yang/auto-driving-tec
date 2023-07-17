#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(LaneDetect)

#boost
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#opencv
find_package(OpenCV)
include_directories(${OpenCV})

#src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
aux_source_directory(${SRC_DIR} SRC)

find_package(Threads)


#main
add_executable(LaneDetector ${CMAKE_CURRENT_SOURCE_DIR}/LaneDetector.cpp ${SRC} ${COMMON})
target_include_directories(LaneDetector PRIVATE SRC_DIR)
target_link_libraries(LaneDetector ${Boost_LIBRARIES} ${OpenCV_LIBS} pthread)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

option(SHOW_DETAIL "show_detail")
set(SHOW_DETAIL OFF)
message(${SHOW_DETAIL})
if(SHOW_DETAIL)
    add_definitions(-DSHOW_DETAIL)
    message(("add_definitions SHOW_DETAIL:" ${SHOW_DETAIL}))
else ()
    message(("add_definitions SHOW_DETAIL:" ${SHOW_DETAIL}))
endif()
