#include <fstream>
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "base.h"

void pcd2bin(const std::string &in_filePath, const std::string &out_filePath)
{

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    if (pcl::io::loadPCDFile<pcl::PointXYZI>(in_filePath, *cloud) == -1)
    {
        std::string err = "Couldn't read file " + in_filePath;
        PCL_ERROR(err.c_str());
        return;
    }

    std::ofstream output(out_filePath.c_str(), std::ios::out | std::ios::binary);
    float intensity = 1.0;
    for (int j = 0; j < cloud->size(); j++)
    {
        output.write((char *)&cloud->at(j).x, sizeof(cloud->at(j).x));
        output.write((char *)&cloud->at(j).y, sizeof(cloud->at(j).y));
        output.write((char *)&cloud->at(j).z, sizeof(cloud->at(j).z));
        output.write((char *)&cloud->at(j).intensity, sizeof(cloud->at(j).intensity));
    }
    output.close();

    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " points from "
              << in_filePath
              << ", writing to "
              << out_filePath
              << std::endl;
}

int main(int argc, char **argv)
{
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
        return 0;
    }
    // 获取文件目录
    std::string in_dirPath(argv[1]);
    std::cout << "input dir path: " << argv[1] << std::endl;
    // 输出文件目录
    // 如：in_dirPath = "/home/user/data/kitti/"
    // out_dirPath = "/home/user/data/pcd/"
    std::vector<std::string> str_vec;
    boost::split(str_vec, in_dirPath, boost::is_any_of("/"), boost::token_compress_on);
    std::string out_dirPath{};
    if (str_vec.size() > 0)
    {
        int i{};
        if (str_vec[0] != "")
            out_dirPath += str_vec[0];
        for (int ind = 1; ind < str_vec.size() - 2; ++ind)
        {
            out_dirPath += "/" + str_vec[ind];
        }
    }
    out_dirPath = out_dirPath + "/" + "bin/";
    if (!mkDir(out_dirPath))
    {
        std::cout << "failed to mkdir,the path is: " << out_dirPath << std::endl;
        return 0;
    }
    std::vector<boost::filesystem::path> stream = getFiles(in_dirPath, "pcd");

    for (auto ite = stream.begin(); ite != stream.end(); ++ite)
    {
        std::string out_file = out_dirPath + (*ite).stem().string() + ".bin";
        pcd2bin((*ite).string(), out_file);
    }
    std::cout << "================" << std::endl
              << "[Done] OK!" << std::endl
              << "================" << std::endl;
}