#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

struct PointXYZIT
{
    float x;
    float y;
    float z;
    unsigned char intensity;
    double timestamp;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment
POINT_CLOUD_REGISTER_POINT_STRUCT(
    PointXYZIT,
    (float, x, x)(float, y, y)(float, z, z)(std::uint8_t, intensity,
                                            intensity)(double, timestamp,
                                                       timestamp))

int main(int argc, char **argv)
{

    std::string intput_pcd_file_path(argv[1]);
    std::string output_pcd_file_path(argv[2]);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<PointXYZIT>::Ptr cloud(new pcl::PointCloud<PointXYZIT>);
    pcl::io::loadPCDFile(intput_pcd_file_path, *cloud);
    for (unsigned int item = 0; item < cloud->size(); ++item)
    {
        pcl::PointXYZI point;
        if ((*cloud)[item].z > 2)
            continue;
        point.x = (*cloud)[item].x;
        point.y = (*cloud)[item].y;
        point.z = (*cloud)[item].z;
        point.intensity = (*cloud)[item].intensity;
        cloud_ptr->push_back(point);
    }
    pcl::io::savePCDFileBinaryCompressed(output_pcd_file_path, *cloud_ptr);
}