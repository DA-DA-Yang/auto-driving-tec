
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>

#include "common/define.h"
class BoundingBox
{

public:
    BoundingBox(/* args */);
    ~BoundingBox();

    bool getBbox(const std::vector<PointXYZI> &pointcloud, BOX_PCL &box);

    bool buildPolygon(const std::vector<PointXYZI> &pointcloud);
    inline const Eigen::Vector3f &getCloudCenter() { return _cloud_center; }
    inline const Eigen::Vector3f &getBboxSize() { return _box_size; }
    inline const std::vector<PointXYZ>& getPolygon() { return _polygon; }
    inline float getAngleYaw() { return _angle_yaw; }

private:
    void _computePolygon2D(std::vector<PointXYZI> &pointcloud);
    void _getMinMax3D(const std::vector<PointXYZI> &pointcloud, Eigen::Vector3f *min_pt, Eigen::Vector3f *max_pt);
    void _setDefaultValue(Eigen::Vector3f &min_pt, Eigen::Vector3f &max_pt);
    bool _linePerturbation(std::vector<PointXYZI> &pointcloud);
    void _computePolygonSizeAndCenter(std::vector<PointXYZ>& polygon);
    void _computeBboxAlongOneEdge(std::vector<PointXYZ> &polygon, const int p1_index, const int p2_index,
                                  Eigen::Vector3d *center, double *length, double *width, double *area, Eigen::Vector3d *direction);
    void _computePolygonSizeAndCenter();
    void _computeBboxSizeAndCenter2D(const std::vector<PointXYZI> &pointcloud, const Eigen::Vector3f &dir, 
                                     Eigen::Vector3f *size, Eigen::Vector3f *center);

private:
    Eigen::Vector3f _cloud_center;
    Eigen::Vector3f _box_size;
    Eigen::Vector3f _direction;
    float _angle_yaw;
    std::vector<PointXYZI> _pointcloud;
    std::vector<PointXYZ> _polygon;
    Eigen::Vector3f _min_pt;
    Eigen::Vector3f _max_pt;
};
