
#include <fstream>
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "base.h"

void bin2pcd(const std::string &in_filePath, const std::string &out_filePath)
{
  // load point cloud
  std::fstream input(in_filePath.data(), std::ios::in | std::ios::binary);
  if (!input.good())
  {
    std::cerr << "Could not read file: " << in_filePath << std::endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, std::ios::beg);

  pcl::PointCloud<pcl::PointXYZI>::Ptr points(
      new pcl::PointCloud<pcl::PointXYZI>);

  int i{};
  for (i = 0; input.good() && !input.eof(); i++)
  {
    pcl::PointXYZI point;
    input.read((char *)&point.x, 3 * sizeof(float));
    input.read((char *)&point.intensity, sizeof(float));
    points->push_back(point);
  }
  input.close();

  std::cout << "Read " << i << " points, writing to "
            << out_filePath << std::endl;
  pcl::PCDWriter writer;

  // Save DoN features
  writer.write<pcl::PointXYZI>(out_filePath, *points);
  // pcl::io::savePCDFileASCII(out_file, *points);
}

int main(int argc, char **argv)
{
  if (argc <= 1)
  {
    std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
    return 0;
  }
  // 获取文件目录
  std::string bin_dirPath(argv[1]);
  std::cout << "bin path: " << argv[1] << std::endl;
  // 输出文件目录
  // 如：in_dirPath = "/home/user/data/kitti/"
  // out_dirPath = "/home/user/data/pcd/"
  std::vector<std::string> str_vec;
  boost::split(str_vec, bin_dirPath, boost::is_any_of("/"), boost::token_compress_on);
  std::string pcd_dirPath{};
  if(str_vec.size()>0)
  {
    int i{};
    if (str_vec[0] != "")
      pcd_dirPath += str_vec[0];
    for (int ind = 1; ind < str_vec.size() - 2; ++ind)
    {
      pcd_dirPath += "/" + str_vec[ind];
    }
  }
  pcd_dirPath = pcd_dirPath + "/" + "pcd/";
  if (!mkDir(pcd_dirPath))
  {
    std::cout << "failed to mkdir,the path is: " << pcd_dirPath << std::endl;
    return 0;
  }
  std::vector<boost::filesystem::path> stream = getFiles(bin_dirPath, "bin");

  for (auto ite = stream.begin(); ite != stream.end(); ++ite)
  {
    std::string out_file = pcd_dirPath + (*ite).stem().string() + ".pcd";
    bin2pcd((*ite).string(), out_file);
  }
  std::cout << "================" << std::endl
            << "[Done] OK!" << std::endl
            << "================" << std::endl;
}