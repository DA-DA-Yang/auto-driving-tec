#include "common/define.h"
#include "perception/common/i_lib/pc/i_ground.h"

class GroundDetector
{

public:
    GroundDetector(/* args */);
    ~GroundDetector();

    bool detect(const PointXYZ &cloud_center, std::vector<PointXYZI>& pointcloud, std::vector<int>& non_ground_point_indices);

private:
    common::PlaneFitGroundDetectorParam* _pfgd_param = nullptr;
    common::PlaneFitGroundDetector* _pfgd = nullptr;
    float _ground_height_threshold = 0.25f;
    Eigen::Vector3d _cloud_center = Eigen::Vector3d(0.0, 0.0, 0.0);
    std::vector<float> _cloud_data;
    std::vector<float> _ground_height_new;
    std::vector<int> _point_indices_temp;
};
