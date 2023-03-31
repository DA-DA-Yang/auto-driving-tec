#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef Eigen::Array<float, 3, 1> Array3f;
typedef Eigen::Array<float, 7, 1> Array7f;
typedef Eigen::Array<int, 2, 1> Array2i;
typedef Eigen::Array<int, 3, 1> Array3i;

typedef Eigen::Matrix<float, Eigen::Dynamic, 4> MatrixX4f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 32> MatrixX32f;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3> MatrixX3i;

// const Array3f pc_range_high(35.2, 40, 1);
const Array3f pc_range_high(70.4, 40, 1);
// const Array3f pc_range_low(-35.2, -40, -3);
const Array3f pc_range_low(0, -40, -3);
const Array3f voxel_size(0.2, 0.2, 4);

const int max_points_per_voxel = 50;
const int max_voxels = 20000; // 12000
const int num_in_features = 4;
// const int num_out_features = 32;
const int num_out_features = 10;
const int num_hidden_features = 32;

// 3D-Box
struct BOX_PCL
{
    // 三维坐标
    float x;
    float y;
    float z;
    // 三维尺寸
    float l;
    float w;
    float h;
    // 方向角
    float r;
    // 类别序号
    int n;
    // 类别名称
    std::string label;
    // 颜色
    pcl::RGB color;
};

// 类别序号与类别名称的映射
std::map<int, std::string> NUMBER_LABEL_MAP{
    {0, std::string("Car")},
    {1, std::string("Pedestrian")},
    {2, std::string("Cyclist")},
    {3, std::string("VAN")}};

// 类别与显示颜色的映射
std::map<std::string, pcl::RGB> LABEL_COLOR_MAP{
    {std::string("Car"), pcl::RGB{255, 0, 0}},
    {std::string("Pedestrian"), pcl::RGB{0, 0, 255}},
    {std::string("Cyclist"), pcl::RGB{0, 255, 0}},
    {std::string("VAN"), pcl::RGB{180, 180, 180}}};

// a1000:label type
enum LABEL_TYPE
{
    Car,
    Pedestrian,
    Cyclist,
    VAN,
};

struct PointXYZI
{
    float X;
    float Y;
    float Z;
    float I;
};