#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(point_pillars)

#boost
find_package(Boost REQUIRED COMPONENTS thread)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#point_pillar src
set(POINTPILLARS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
aux_source_directory(${POINTPILLARS_SRC_DIR} POINTPILLARS_SRC)

#common
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../common)
aux_source_directory(${COMMON_DIR} COMMON)

#pfe
add_executable(Pfe ${CMAKE_CURRENT_SOURCE_DIR}/pfe.cpp ${POINTPILLARS_SRC} ${COMMON})
target_include_directories(Pfe PRIVATE POINTPILLARS_SRC_DIR)
target_link_libraries(Pfe ${PCL_LIBRARIES} ${Boost_THREAD_LIBRARY})

#post
add_executable(Post ${CMAKE_CURRENT_SOURCE_DIR}/post.cpp ${POINTPILLARS_SRC} ${COMMON})
target_include_directories(Post PRIVATE POINTPILLARS_SRC_DIR)
target_link_libraries(Post ${PCL_LIBRARIES} ${Boost_THREAD_LIBRARY})

