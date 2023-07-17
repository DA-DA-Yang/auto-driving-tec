#include <localization/ndt/ndt_omp/gicp_omp.h>
#include <localization/ndt/ndt_omp/gicp_omp_impl.hpp>

template class pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ>;
template class pclomp::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI>;

