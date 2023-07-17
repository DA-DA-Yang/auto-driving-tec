
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include "gflags/gflags.h"

#include "localization/ndt/src/async_buffer.h"
#include "localization/ndt/common/ndt_flags.h"

#include "localization/ndt/ndt_omp/ndt_omp.h"

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = pcl::PointCloud<Point>::Ptr;
using PointCloudConstPtr = const pcl::PointCloud<Point>::Ptr;

// the whole map
PointCloudPtr map_ptr(new PointCloud());
PointCloudPtr align_map_ptr(new PointCloud());
std::vector<PointCloud> align_map_vec;
std::vector<PointCloud> show_map_vec;
bool is_first_map = true;
unsigned int frame_index = 0;
unsigned int last_frame_index = 0;
unsigned int last_map_index = 0;
unsigned int show_count = 0;
bool stop_show = false;

// runtime pose
Eigen::Affine3d added_pose = Eigen::Affine3d::Identity();
Eigen::Affine3d current_pose = Eigen::Affine3d::Identity();
Eigen::Affine3d previous_pose = Eigen::Affine3d::Identity();
Eigen::Affine3d diff_pose = Eigen::Affine3d::Identity();
// lidar pose
std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> pcd_poses;
std::vector<double> time_stamps;
std::vector<unsigned int> pcd_indices;

// ndt_omp
pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp;

std::unique_ptr<AsyncBuffer> async_buffer_ptr;

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Cloud Map"));

void Init()
{
    // ndt_omp
    ndt_omp.setTransformationEpsilon(FLAGS_trans_eps);
    ndt_omp.setStepSize(FLAGS_step_size);
    ndt_omp.setResolution(static_cast<float>(FLAGS_ndt_res));
    ndt_omp.setMaximumIterations(FLAGS_max_iter);
    int num_threads = omp_get_max_threads();
    pclomp::NeighborSearchMethod search_method = pclomp::KDTREE;
    ndt_omp.setNumThreads(num_threads);
    ndt_omp.setNeighborhoodSearchMethod(search_method);
    std::cout << "openmp threads: " << num_threads << std::endl;
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
    std::cout << "pcd_poses: " << pcd_poses.size() << std::endl
              << "pcd indices: " << pcd_indices.size() << std::endl
              << "pcd file size: " << file_paths.size() << std::endl;

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

void VoxelFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr input, pcl::PointCloud<pcl::PointXYZ>::Ptr output)
{
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid_filter;
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

    // 范围滤波
    PointCloudPtr scan_ptr(new PointCloud());
    RangeFilter(cloud_ptr, scan_ptr);
    // 是否为初始帧
    if (is_first_map)
    {
        // 第一帧直接加入地图
        *map_ptr += *scan_ptr;
        *align_map_ptr += *scan_ptr;
        align_map_vec.push_back(*scan_ptr);
        is_first_map = false;
        return;
    }
    // 体素滤波
    PointCloudPtr voxel_ptr(new PointCloud());
    VoxelFilter(scan_ptr, voxel_ptr);
    // target与source点云格式化转换
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    int num_target_points = static_cast<int>(align_map_ptr->size());
    int num_source_points = static_cast<int>(voxel_ptr->size());

    for (int i = 0; i < num_target_points; ++i)
    {
        pcl::PointXYZ pt{(*align_map_ptr)[i].x, (*align_map_ptr)[i].y, (*align_map_ptr)[i].z};
        target_cloud->push_back(pt);
    }
    for (int i = 0; i < num_source_points; ++i)
    {

        pcl::PointXYZ pt{(*voxel_ptr)[i].x, (*voxel_ptr)[i].y, (*voxel_ptr)[i].z};
        source_cloud->push_back(pt);
    }
    // 设置当前点云
    ndt_omp.setInputSource(source_cloud);
    std::cout << "source size:" << source_cloud->size() << std::endl;
    // 设置地图点云
    ndt_omp.setInputTarget(target_cloud);
    std::cout << "target size:" << target_cloud->size() << std::endl;

    // 获得前后两帧之间的相对位姿，结合上帧结果作为初值估计，这种方法最好。
    // 因为室内没有GPS信号，因此平移向量直接设为上帧的平移
    Eigen::Affine3d relative_pose = Eigen::Affine3d::Identity();
    // relative_pose = GetLidarRelativePose();
    relative_pose.linear() = GetLidarRelativePose().linear();
    previous_pose.translation() += diff_pose.translation();
    Eigen::Affine3d ini_trans = previous_pose * relative_pose;
    Eigen::Matrix4f guess_pose = (previous_pose * relative_pose).matrix().cast<float>();

    // ndt配准
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    ndt_omp.align(*output_cloud, guess_pose);

    // 变换矩阵
    Eigen::Matrix4f t_localizer = ndt_omp.getFinalTransformation();
    // 配准得分
    double fitness_score = ndt_omp.getFitnessScore();
    // 是否收敛
    bool has_converged = ndt_omp.hasConverged();
    // 迭代次数
    int final_num_iteration = ndt_omp.getFinalNumIteration();
    // 变换概率
    double transformation_probability = ndt_omp.getTransformationProbability();

    // 当前帧的位姿(相对于初始帧)
    current_pose = t_localizer.cast<double>();
    // 相邻帧平移向量的差值
    diff_pose.translation() = current_pose.translation() - previous_pose.translation();
    // 保存为历史位姿
    previous_pose = current_pose;

    // 距离上次地图保存时刻的位姿距离，每隔3m更新一次地图
    double shift = SquaredDistance(current_pose, added_pose);
    if (shift >= FLAGS_min_add_scan_shift)
    {
        // 当前点云转换到地图坐标系下，并加入到地图点云中
        Eigen::Affine3d transform = current_pose;
        PointCloudPtr transformed_scan_map(new PointCloud());
        pcl::transformPointCloud(*scan_ptr, *transformed_scan_map, transform);
        // 地图点云
        *map_ptr += *transformed_scan_map;
        *align_map_ptr += *transformed_scan_map;
        // 用于地图显示
        align_map_vec.push_back(*transformed_scan_map);
        std::cout << std::endl
                  << "map add : Frame " << pcd_indices[frame_index] << std::endl;
        added_pose = current_pose;

        // 防止用于配准的点云过大，最多保存10帧地图点云用于配准
    }

    std::cout << "score: " << fitness_score
              << " converged: " << has_converged
              << " iteration: " << final_num_iteration
              << " probability: " << transformation_probability
              << " shift: " << shift
              << std::endl;

    // 左乘变换点的坐标，右乘变换坐标系(获得点在新左坐标系的表达)
    // 获得初始帧的位姿
    Eigen::Affine3d trans_first = Eigen::Affine3d::Identity();
    trans_first = pcd_poses[0];
    // 查看与组合导航的差值
    auto xyz_diff = (trans_first * current_pose).translation() - pcd_poses[frame_index].translation();
    std::cout << "dx: " << xyz_diff(0) << " dy: " << xyz_diff(1) << " dz:" << xyz_diff(2) << std::endl;
    Eigen::Quaterniond quaterniond = (Eigen::Quaterniond)pcd_poses[frame_index].linear();
    std::cout << "Ori quaterniond x: " << quaterniond.x()
              << " y: " << quaterniond.y()
              << " z: " << quaterniond.z()
              << " w: " << quaterniond.w()
              << std::endl;
    quaterniond = (Eigen::Quaterniond)(trans_first * current_pose).linear();
    std::cout << "New quaterniond x: " << quaterniond.x()
              << " y: " << quaterniond.y()
              << " z: " << quaterniond.z()
              << " w: " << quaterniond.w()
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
    // 创建输出文件夹
    std::string out_dir = FLAGS_output_dir + "pcd_map";
    int res = access(out_dir.c_str(), F_OK);
    if (res == 0)
    {
        res = system(("rm -rf " + out_dir).c_str());
    }
    res = system(("mkdir " + out_dir).c_str());
    // 保存地图时，不进行平移变换，保证初始帧的中心点为原点
    std::string map_file = out_dir + "/1.pcd";
    Eigen::Affine3d align_pose = Eigen::Affine3d::Identity();
    align_pose.linear() = pcd_poses[0].linear();
    PointCloudPtr transformed_scan_ptr(new PointCloud());
    pcl::transformPointCloud(*map_ptr, *transformed_scan_ptr, align_pose);
    pcl::io::savePCDFileBinaryCompressed(map_file, *transformed_scan_ptr);
    std::cout << std::endl
              << std::setprecision(6) << "UTM relative coordinates"
              << " x: " << align_pose.translation().x()
              << " y: " << align_pose.translation().y()
              << " z: " << align_pose.translation().z() << std::endl;
    FILE *fp;
    std::string pose_file = out_dir + "/poses.txt";
    fp = fopen(pose_file.c_str(), "w+");
    fprintf(fp, "1 1 1000 1000 1 0 0 0 1");
    fclose(fp);
    std::cout << std::endl
              << "map create success!"
              << std::endl;
}

void showMap()
{
    while (!stop_show)
    {
        if (align_map_vec.size())
        {
            show_map_vec.push_back(align_map_vec[0]);
            align_map_vec.erase(align_map_vec.begin());
            // 添加点云
            std::string name = "cloud" + std::to_string(show_count);
            show_count++;
            // 按"intensity"着色
            // pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(show_map_vec.back().makeShared(), "intensity");
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> fildColor(show_map_vec.back().makeShared(), 200, 200, 200);
            viewer->addPointCloud<pcl::PointXYZI>(show_map_vec.back().makeShared(), fildColor, name);
            // 设置点云大小
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, name);
        }
        viewer->spinOnce(100);
    }
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
    if (event.getKeySym() == "k" && event.keyDown())
    {
        stop_show = !stop_show;
    }
}

int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    auto start_time = std::chrono::system_clock::now();

    if (access(FLAGS_output_dir.c_str(), F_OK) != 0)
    {
        std::cout << "错误：output_dir 不存在。" << std::endl;
        return 0;
    }

    Init();
    LoadPoses();

    StartAsyncReadProcess();

    // lidar-IMU的外参
    Eigen::Translation3d trans(0.23, 0, 1.10);
    Eigen::Quaterniond quat(0.7071068, 0.7071068, 0.0, 0.0);
    static Eigen::Affine3d lidar_extrinsic = trans * quat;

    std::thread show_thread = std::thread(&showMap);

    while (!async_buffer_ptr->IsEnd())
    {
        PointCloudPtr cloud_ptr = async_buffer_ptr->Get();
        if (cloud_ptr)
        {
            LidarProcess(cloud_ptr);
            // createMap(cloud_ptr);
            frame_index++;
            if (pcd_indices[frame_index] > 5000)
                break;
        }
    }

    SaveMap();

    // 配准耗时
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << std::endl
              << "NDT mapping cost:" << diff.count() << " s"
              << std::endl
              << std::endl;
    // 注册键盘响应事件
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&viewer);

    while (!stop_show)
    {
        show_thread.join();
    }
    return 0;
}
