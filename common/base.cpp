#include "base.h"

bool cmp_filePath(boost::filesystem::path int_path1, boost::filesystem::path in_path2)
{
    // 按文件名的数字升序
    return std::stoi(int_path1.stem().string()) < std::stoi(in_path2.stem().string());
}

std::vector<boost::filesystem::path> getFiles(std::string in_dirPath, std::string in_suffix)
{
    std::vector<boost::filesystem::path> paths(boost::filesystem::directory_iterator{in_dirPath}, boost::filesystem::directory_iterator{});
    // extension()返回的后缀带"."
    std::string suf = "." + in_suffix;
    for (auto ite = paths.begin(); ite != paths.end(); ++ite)
    {
        if ((*ite).extension().string() != suf)
        {
            ite = paths.erase(ite);
            ite--;
        }
    }
    sort(paths.begin(), paths.end(), cmp_filePath);
    return paths;
}

void binToPclPtr(std::string in_filePath, pcl::PointCloud<pcl::PointXYZI>::Ptr out_pointPtr)
{
    std::fstream input(in_filePath.data(), std::ios::in | std::ios::binary);
    if (!input.good())
    {
        std::cerr << "Could not read file: " << in_filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);

    out_pointPtr->clear();

    int i{};
    for (i = 0; input.good() && !input.eof(); i++)
    {
        pcl::PointXYZI point;
        input.read((char *)&point.x, 3 * sizeof(float));
        input.read((char *)&point.intensity, sizeof(float));
        out_pointPtr->push_back(point);
    }
    input.close();
}

bool mkDir(std::string in_dirPath)
{
    if(rmDir(in_dirPath))
    {
        auto res = mkdir(in_dirPath.data(), S_IRWXU);
        if(res==0)
            return true;
        else
            return false;
    }
    else
        return false;
}

bool rmDir(std::string in_dirPath)
{
    if (access(in_dirPath.c_str(), F_OK)==0)
    {
        system(("rm -rf " + in_dirPath).c_str());
        if (access(in_dirPath.c_str(), F_OK)==-1)
            return true;
        else
            return false;
    }
    else
    {
        return true;
    }
}