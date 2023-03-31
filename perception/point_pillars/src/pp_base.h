#pragma once

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Eigen>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <numeric>

using namespace std;
using namespace Eigen;

typedef Eigen::Array<float, 3, 1> Array3f;
typedef Eigen::Array<float, 7, 1> Array7f;
typedef Eigen::Array<int, 2, 1> Array2i;
typedef Eigen::Array<int, 3, 1> Array3i;

typedef Eigen::Matrix<float, Eigen::Dynamic, 4> MatrixX4f;
typedef Eigen::Matrix<float, Eigen::Dynamic, 32> MatrixX32f;
typedef Eigen::Matrix<int, Eigen::Dynamic, 3> MatrixX3i;

// const Array3f pc_range_high(35.2, 40, 1);
// const Array3f pc_range_low(-35.2, -40, -3);
const Array3f pc_range_high(70.4, 40.0, 1.0);
const Array3f pc_range_low(0.0, -40.0, -3.0);
const Array3f voxel_size(0.2, 0.2, 4);
const int batch_image_height = int((pc_range_high[1] - pc_range_low[1]) /
                               voxel_size[1]);
const int batch_image_width = int((pc_range_high[0] - pc_range_low[0]) /
                              voxel_size[0]);

const int max_points_per_voxel = 50;
const int max_voxels = 20000; // 12000
const int num_in_features = 4;
const int num_hidden_features = 32;
const int num_out_features = 10;  
const int box_coder_size = 7;

const std::string class_name_Car = "Car";
const Array3f size_Car(1.6, 3.9, 1.56);
const Array3f stride_Car(voxel_size[0] * 2, voxel_size[1] * 2, 0.0);
const Array3f offset_Car(pc_range_low[0] + voxel_size[0],
                         pc_range_low[1] + voxel_size[1], -1.78);

const std::string class_name_Pedestrian = "Pedestrian";
const Array3f size_Pedestrian(0.6, 0.8, 1.73);
const Array3f stride_Pedestrian(voxel_size[0] * 2, voxel_size[1] * 2, 0.0);
const Array3f offset_Pedestrian(pc_range_low[0] + voxel_size[0],
                                pc_range_low[1] + voxel_size[1], -1.465);

const std::string class_name_Cyclist = "Cyclist";
const Array3f size_Cyclist(0.6, 1.76, 1.73);
const Array3f stride_Cyclist(voxel_size[0] * 2, voxel_size[1] * 2, 0.0);
const Array3f offset_Cyclist(pc_range_low[0] + voxel_size[0],
                             pc_range_low[1] + voxel_size[1], -1.465);

const std::string class_name_Van = "Van";
const Array3f size_Van(1.87103749, 5.02808195, 2.20964255);
const Array3f stride_Van(voxel_size[0] * 2, voxel_size[1] * 2, 0.0);
const Array3f offset_Van(pc_range_low[0] + voxel_size[0],
                         pc_range_low[1] + voxel_size[1], -1.78);

// const float nms_score_threshold = 0.45;
const float nms_score_threshold = -0.20067069546215122; // argsigmoid
const float nms_iou_threshold = 0.3;
const int num_pre_max_size = 100;

