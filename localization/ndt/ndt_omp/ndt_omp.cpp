#include <localization/ndt/ndt_omp/ndt_omp.h>
#include <localization/ndt/ndt_omp/ndt_omp_impl.hpp>

template class pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ>;
template class pclomp::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI>;
