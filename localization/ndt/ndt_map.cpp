/*
 * Copyright (c) 2021 daohu527 <daohu527@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 *  Copyright (c) 2015, Nagoya University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither the name of Autoware nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 *  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>

#include "gflags/gflags.h"

#include "localization/ndt/src/async_buffer.h"
#include "localization/ndt/common/ndt_flags.h"

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = pcl::PointCloud<Point>::Ptr;
using PointCloudConstPtr = const pcl::PointCloud<Point>::Ptr;

// the whole map
PointCloudPtr map_ptr(new PointCloud());
PointCloudPtr align_map_ptr(new PointCloud());
std::vector<PointCloud> align_map_vec;
bool is_first_map = true;
unsigned int frame_index = 0;

// runtime pose
Eigen::Affine3d added_pose = Eigen::Affine3d::Identity();
Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
Eigen::Affine3d previous_pose = Eigen::Affine3d::Identity();

// lidar pose
std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> pcd_poses;
std::vector<double> time_stamps;
std::vector<unsigned int> pcd_indices;

// ndt
pcl::NormalDistributionsTransform<Point, Point> ndt;

//
std::unique_ptr<AsyncBuffer> async_buffer_ptr;

void Init()
{
  // ndt
  ndt.setTransformationEpsilon(FLAGS_trans_eps);
  ndt.setStepSize(FLAGS_step_size);
  ndt.setResolution(static_cast<float>(FLAGS_ndt_res));
  ndt.setMaximumIterations(FLAGS_max_iter);
}

void LoadPcdPoses(const std::string &file_path,
                  std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> *poses,
                  std::vector<double> *timestamps,
                  std::vector<unsigned int> *pcd_indices)
{
  poses->clear();
  timestamps->clear();
  pcd_indices->clear();

  FILE *file = fopen(file_path.c_str(), "r");
  if (file)
  {
    unsigned int index;
    double timestamp;
    double x, y, z;
    double qx, qy, qz, qr;
    double last_timestamp = 0;
    static constexpr int kSize = 9;
    while (fscanf(file, "%u %lf %lf %lf %lf %lf %lf %lf %lf\n", &index,
                  &timestamp, &x, &y, &z, &qx, &qy, &qz, &qr) == kSize)
    {
      if ((timestamp - last_timestamp) < FLAGS_sample_interval_time)
        continue;
      Eigen::Translation3d trans(Eigen::Vector3d(x, y, z));
      Eigen::Quaterniond quat(qr, qx, qy, qz);
      poses->push_back(trans * quat);
      timestamps->push_back(timestamp);
      pcd_indices->push_back(index);
      last_timestamp = timestamp;
    }
    fclose(file);
  }
  else
  {
    std::cout << "Can't open file to read: " << file_path << std::endl;
  }
}

// load all poses from file
void LoadPoses()
{
  std::string pose_file = FLAGS_workspace_dir + "/poses.txt";

  LoadPcdPoses(pose_file, &pcd_poses, &time_stamps, &pcd_indices);

  std::cout << "pcd_poses: " << pcd_poses.size()
            << " ,pcd_indices: " << pcd_indices.size() << std::endl;
}

void StartAsyncReadProcess()
{
  std::vector<std::string> file_paths;
  for (size_t i = 0; i < pcd_indices.size(); ++i)
  {
    unsigned int index = pcd_indices[i];
    std::ostringstream ss;
    ss << FLAGS_workspace_dir << "/" << index << ".pcd";
    std::string pcd_file_path = ss.str();
    if (access(pcd_file_path.c_str(), F_OK) == 0)
      file_paths.push_back(pcd_file_path);
    else
    {
      pcd_poses.erase(pcd_poses.begin() + i);
      pcd_indices.erase(pcd_indices.begin() + i);
      i--;
    }
  }

  std::cout << "Pcd file size: " << file_paths.size() << std::endl;

  async_buffer_ptr.reset(new AsyncBuffer(file_paths));
  async_buffer_ptr->Init();
}

void RangeFilter(PointCloudConstPtr input, PointCloudPtr output)
{
  for (PointCloud::const_iterator item = input->begin(); item != input->end();
       item++)
  {
    Point point;
    point.x = item->x;
    point.y = item->y;
    point.z = item->z;
    point.intensity = item->intensity;

    double r = std::pow(point.x, 2.0) + std::pow(point.y, 2.0);
    if (FLAGS_min_scan_range <= r && r <= FLAGS_max_scan_range)
    {
      output->push_back(point);
    }
  }
}

void VoxelFilter(PointCloudConstPtr input, PointCloudPtr output)
{
  pcl::VoxelGrid<Point> voxel_grid_filter;
  float voxel_leaf_size = static_cast<float>(FLAGS_voxel_leaf_size);
  voxel_grid_filter.setLeafSize(voxel_leaf_size, voxel_leaf_size,
                                voxel_leaf_size);
  voxel_grid_filter.setInputCloud(input);
  voxel_grid_filter.filter(*output);
}

Eigen::Affine3d GetLidarRelativePose()
{
  static unsigned int index = 1;
  Eigen::Affine3d relative_pose;
  if (frame_index < pcd_poses.size())
  {
    relative_pose = pcd_poses[frame_index - 1].inverse() * pcd_poses[frame_index];
  }
  return relative_pose;
}

double SquaredDistance(Eigen::Affine3d first, Eigen::Affine3d second)
{
  Eigen::Translation3d first_transd(first.translation());
  Eigen::Translation3d second_transd(second.translation());

  return std::pow(first_transd.x() - second_transd.x(), 2.0) + std::pow(first_transd.y() - second_transd.y(), 2.0);
}

void LidarProcess(PointCloudPtr cloud_ptr)
{
  std::cout << std::endl
            << "Frame: " << pcd_indices[frame_index] << std::endl;

  PointCloudPtr scan_ptr(new PointCloud());
  RangeFilter(cloud_ptr, scan_ptr);

  // When creating the map for the first time,
  // get the first frame and add it to the map, then return
  if (is_first_map)
  {
    *map_ptr += *scan_ptr;
    *align_map_ptr += *scan_ptr;
    align_map_vec.clear();
    align_map_vec.push_back(*scan_ptr);
    is_first_map = false;
    return;
  }

  PointCloudPtr voxel_ptr(new PointCloud());
  VoxelFilter(scan_ptr, voxel_ptr);
  // 设置当前点云
  ndt.setInputSource(voxel_ptr);
  // 设置点云地图
  ndt.setInputTarget(align_map_ptr);
  // ndt.setInputTarget(map_ptr);

  // Get the relative pose between 2 frames
  // 获得两帧之间的相对位姿
  Eigen::Affine3d relative_pose = GetLidarRelativePose();
  // Calculate the guess pose based on the position of the previous frame
  Eigen::Matrix4f guess_pose =
      (previous_pose * relative_pose).matrix().cast<float>();
  // Eigen::Matrix4f guess_pose =
  //     (previous_pose).matrix().cast<float>();

  // Eigen::Affine3d tmp = Eigen::Affine3d::Identity();
  // tmp = guess_pose.cast<double>();
  // // translation
  // Eigen::Quaterniond quaterniond = (Eigen::Quaterniond)relative_pose.linear();
  // std::cout << std::setprecision(6) << "relative translation"
  //           << " x: " << relative_pose.translation().x()
  //           << " y: " << relative_pose.translation().y()
  //           << " z: " << relative_pose.translation().z() << std::endl;
  // quaterniond = (Eigen::Quaterniond)relative_pose.linear();
  // std::cout << std::setprecision(6) << "previous translation"
  //           << " x: " << previous_pose.translation().x()
  //           << " y: " << previous_pose.translation().y()
  //           << " z: " << previous_pose.translation().z() << std::endl;
  // std::cout << std::setprecision(6) << "   guess translation"
  //           << " x: " << tmp.translation().x()
  //           << " y: " << tmp.translation().y()
  //           << " z: " << tmp.translation().z() << std::endl;
  // // raotation
  // std::cout << "relative quaterniond x: " << quaterniond.x()
  //           << " y: " << quaterniond.y()
  //           << " z: " << quaterniond.z()
  //           << " w: " << quaterniond.w()
  //           << std::endl;
  // quaterniond = (Eigen::Quaterniond)previous_pose.linear();
  // std::cout << "previous quaterniond x: " << quaterniond.x()
  //           << " y: " << quaterniond.y()
  //           << " z: " << quaterniond.z()
  //           << " w: " << quaterniond.w()
  //           << std::endl;
  // quaterniond = (Eigen::Quaterniond)tmp.linear();
  // std::cout << "   guess quaterniond x: " << quaterniond.x()
  //           << " y: " << quaterniond.y()
  //           << " z: " << quaterniond.z()
  //           << " w: " << quaterniond.w()
  //           << std::endl;

  // 使用上一帧的结果用作当前帧的初值估计
  // guess_pose = (previous_pose).matrix().cast<float>();

  PointCloudPtr output_cloud(new PointCloud());
  ndt.align(*output_cloud, guess_pose);

  double fitness_score = ndt.getFitnessScore();
  Eigen::Matrix4f t_localizer = ndt.getFinalTransformation();
  bool has_converged = ndt.hasConverged();
  int final_num_iteration = ndt.getFinalNumIteration();
  double transformation_probability = ndt.getTransformationProbability();

  current_pose = t_localizer.cast<double>();
  previous_pose = current_pose;

  double shift = SquaredDistance(current_pose, added_pose);
  if (shift >= FLAGS_min_add_scan_shift)
  {
    PointCloudPtr transformed_scan_ptr(new PointCloud());
    pcl::transformPointCloud(*scan_ptr, *transformed_scan_ptr, t_localizer);
    *map_ptr += *transformed_scan_ptr;
    added_pose = current_pose;
    std::cout << std::endl
              << "map add : Frame " << pcd_indices[frame_index] << std::endl;

    // 建立用于地图匹配的缓存，避免使用整个地图进行匹配
    if (align_map_vec.size() > FLAGS_align_map_count)
      align_map_vec.erase(align_map_vec.begin());
    align_map_vec.push_back(*transformed_scan_ptr);
    for (size_t i = 0; i < align_map_vec.size(); ++i)
    {
      align_map_ptr->clear();
      *align_map_ptr += align_map_vec[i];
    }
  }

  std::cout << "score: " << fitness_score
            << " converged: " << has_converged
            << " iteration: " << final_num_iteration
            << " probability: " << transformation_probability
            << " shift: " << shift
            << std::endl;
}

void createMap(PointCloudPtr cloud_ptr)
{
  std::cout << std::endl
            << "Frame: " << pcd_indices[frame_index] << std::endl;
  PointCloudPtr transformed_scan_ptr(new PointCloud());
  pcl::transformPointCloud(*cloud_ptr, *transformed_scan_ptr, pcd_poses[frame_index]);
  *map_ptr += *transformed_scan_ptr;
}

void SaveMap()
{
  // // Align coordinates.
  // // The initial coordinates are the pose of the first frame
  // // CHECK(pcd_poses.size() != 0) << "pcd pose is empty";
  // Eigen::Affine3d init_pose = pcd_poses[0];
  // Eigen::Affine3f align_pose = Eigen::Affine3f::Identity();
  // align_pose.linear() = init_pose.cast<float>().linear();

  // std::cout << "Align matrix: " << align_pose.matrix() << std::endl;

  // // Quaterniond
  // Eigen::Quaterniond quaterniond = (Eigen::Quaterniond)init_pose.linear();
  // std::cout << "Align quaterniond x: " << quaterniond.x()
  //           << " y: " << quaterniond.y()
  //           << " z: " << quaterniond.z()
  //           << " w: " << quaterniond.w()
  //           << std::endl;

  // // Rotation
  // auto euler = quaterniond.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
  // std::cout << "Align rotation roll, pitch, yaw " << euler << std::endl;

  // // CHECK(map_ptr != nullptr) << "map is null";
  // PointCloudPtr align_map_ptr(new PointCloud());
  // pcl::transformPointCloud(*map_ptr, *align_map_ptr, align_pose.matrix());

  // std::cout << std::setprecision(15) << "UTM relative coordinates"
  //           << " x: " << init_pose.translation().x()
  //           << " y: " << init_pose.translation().y()
  //           << " z: " << init_pose.translation().z() << std::endl;

  // // Save map
  // pcl::io::savePCDFileBinaryCompressed(FLAGS_output_file, *align_map_ptr);

  PointCloudPtr transformed_scan_ptr(new PointCloud());
  Eigen::Affine3d transform = Eigen::Affine3d::Identity();
  transform.translation() = -pcd_poses[0].translation();
  pcl::transformPointCloud(*map_ptr, *transformed_scan_ptr, transform);
  pcl::io::savePCDFileBinaryCompressed(FLAGS_output_file, *transformed_scan_ptr);
  std::cout << std::endl
            << std::setprecision(6) << "UTM relative coordinates"
            << " x: " << transform.translation().x()
            << " y: " << transform.translation().y()
            << " z: " << transform.translation().z() << std::endl;
}

int main(int argc, char **argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  auto start_time = std::chrono::system_clock::now();

  Init();
  LoadPoses();

  StartAsyncReadProcess();

  while (!async_buffer_ptr->IsEnd())
  {
    PointCloudPtr cloud_ptr = async_buffer_ptr->Get();
    if (cloud_ptr)
    {
      // LidarProcess(cloud_ptr);
      createMap(cloud_ptr);
      frame_index++;
      if (pcd_indices[frame_index] > 5000)
        break;
    }
  }

  // 这里仅保存融合的点云，不作变换，所有点云处于第一帧的坐标系下
  // pcl::io::savePCDFileBinaryCompressed(FLAGS_output_file, *map_ptr);
  SaveMap();

  // Performance
  auto end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end_time - start_time;
  std::cout << std::endl

            << "NDT mapping cost:" << diff.count() << " s"
            << std::endl
            << std::endl;
}
