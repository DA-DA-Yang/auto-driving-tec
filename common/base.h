#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sys/stat.h> //mkdir()
#include <unistd.h> //access()

// 用于sort函数的排序规则
bool cmp_filePath(boost::filesystem::path int_path1, boost::filesystem::path in_path2);

// 从文件夹中获得指定类型的所有文件
std::vector<boost::filesystem::path> getFiles(std::string in_dirPath, std::string in_suffix);

// 从bin文件中读取点云
void binToPclPtr(std::string in_filePath, pcl::PointCloud<pcl::PointXYZI>::Ptr out_pointPtr);

//创建文件夹
bool mkDir(std::string in_dirPath);
//删除文件夹
bool rmDir(std::string in_dirPath);