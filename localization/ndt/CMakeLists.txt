#设置CMake最低版本号
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)
#设置项目名称
project(ndt)

find_package(OpenMP)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "-std=c++0x -fopenmp")
# 解决Eigen问题
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -O3 -mtune=native ")

#pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#src
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
aux_source_directory(${SRC_DIR} SRC)

#ndt_omp
set(NDT_OMP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ndt_omp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ndt_omp)
aux_source_directory(${NDT_OMP_DIR} NDT_OMP)
add_library(ndt_omp ${NDT_OMP_DIR}/voxel_grid_covariance_omp.cpp ${NDT_OMP_DIR}/ndt_omp.cpp ${NDT_OMP_DIR}/gicp_omp.cpp)
target_link_libraries(ndt_omp ${PCL_LIBRARIES})
if(OpenMP_CXX_FOUND)
  target_link_libraries(ndt_omp OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found")
endif()

#common
set(COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)
aux_source_directory(${SRC_DIR} COMMON)
add_library(ndt_flags SHARED ${COMMON_DIR}/ndt_flags.cc)

find_package(Threads)

find_package (gflags REQUIRED)

#main
add_executable(ndt_map ${CMAKE_CURRENT_SOURCE_DIR}/ndt_map.cpp ${SRC})
target_include_directories(ndt_map PRIVATE SRC_DIR)
target_link_libraries(ndt_map ${PCL_LIBRARIES} pthread gflags ndt_flags)

add_executable(ndt_mapping ${CMAKE_CURRENT_SOURCE_DIR}/ndt_mapping.cpp ${SRC})
target_include_directories(ndt_mapping PRIVATE SRC_DIR)
target_link_libraries(ndt_mapping ${PCL_LIBRARIES} pthread gflags ndt_flags)

add_executable(modify_cloud ${CMAKE_CURRENT_SOURCE_DIR}/modify_cloud.cpp ${SRC})
target_include_directories(modify_cloud PRIVATE SRC_DIR)
target_link_libraries(modify_cloud ${PCL_LIBRARIES})

add_executable(ndt_map_creator ${CMAKE_CURRENT_SOURCE_DIR}/ndt_map_creator.cpp ${SRC})
target_include_directories(ndt_map_creator PRIVATE SRC_DIR)
target_link_libraries(ndt_map_creator ${PCL_LIBRARIES} pthread gflags ndt_flags ndt_omp)

