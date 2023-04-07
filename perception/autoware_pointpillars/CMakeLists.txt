#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(autoware_pointpillars)

#设置Debug或者Release编译
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_BUILD_TYPE "Release")

#point_pillar src
set(POINTPILLARS_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../point_pillars/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../point_pillars/src)
aux_source_directory(${POINTPILLARS_SRC_DIR} POINTPILLARS_SRC)

#src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}//src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
aux_source_directory(${SRC_DIR} SRC)

#pfe_autoware
add_executable(Pfe_Autoware ${CMAKE_CURRENT_SOURCE_DIR}/pfe_autoware.cpp ${SRC} ${POINTPILLARS_SRC})
target_include_directories(Pfe_Autoware PRIVATE SRC_DIR POINTPILLARS_SRC_DIR)

#post_autoware
add_executable(Post_Autoware ${CMAKE_CURRENT_SOURCE_DIR}/post_autoware.cpp ${SRC} ${POINTPILLARS_SRC})
target_include_directories(Post_Autoware PRIVATE SRC_DIR POINTPILLARS_SRC_DIR)

