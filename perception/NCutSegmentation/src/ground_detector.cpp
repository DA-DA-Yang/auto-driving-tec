#include "ground_detector.h"

GroundDetector::GroundDetector(/* args */)
{
    _pfgd_param = new common::PlaneFitGroundDetectorParam;
    _pfgd_param->roi_region_rad_x = 120.f;
    _pfgd_param->roi_region_rad_y = 120.f;
    _pfgd_param->roi_region_rad_z = 120.f;
    _pfgd_param->nr_grids_coarse = 16;
    _pfgd_param->nr_smooth_iter = 5;

    _pfgd = new common::PlaneFitGroundDetector(*_pfgd_param);
    _pfgd->Init();
}

GroundDetector::~GroundDetector()
{
}

bool GroundDetector::detect(const PointXYZ &cloud_center, std::vector<PointXYZI> &pointcloud, std::vector<int> &non_ground_point_indices)
{
    unsigned int nr_points_element = 3;

    // 获取点云中心坐标
    _cloud_center(0) = cloud_center.x;
    _cloud_center(1) = cloud_center.y;
    _cloud_center(2) = cloud_center.z;

    int num_points = static_cast<int>(pointcloud.size());
    int data_id = 0;
    int count_valid_point = 0;
    _cloud_data.resize(num_points * 3);
    _ground_height_new.resize(num_points);
    _point_indices_temp.resize(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        PointXYZI pt = pointcloud[i];
        _point_indices_temp[count_valid_point++] = i;
        _cloud_data[data_id++] = pt.x - _cloud_center(0);
        _cloud_data[data_id++] = pt.y - _cloud_center(1);
        _cloud_data[data_id++] = pt.z - _cloud_center(2);
    }

    // 点云平面检测
    bool res = _pfgd->Detect(_cloud_data.data(), _ground_height_new.data(), count_valid_point, nr_points_element);
    if (!res)
    {
        std::cout << "ground_detector.cpp: ground detector failed!" << std::endl;
        return false;
    }

    float z_tmp = 0.f;
    non_ground_point_indices.clear();
    // 给点云重新赋值去除地面后的高度
    for (int i = 0; i < count_valid_point; ++i)
    {
        z_tmp = _ground_height_new[i];
        PointXYZI &pt = pointcloud[i];
        pt.z = z_tmp;
        if (common::IAbs(z_tmp) > _ground_height_threshold)
        {
            non_ground_point_indices.push_back(i);
        }
    }

    return true;
}