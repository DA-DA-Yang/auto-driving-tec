#include "pfe_process.h"
#include "base.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    printf("\n");
    printf("\n>========pre process========\n");
    //该程序用于执行点云pillar化。
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
        return 0;
    }
    // 获取文件目录
    std::string in_dirPath(argv[1]);
    std::cout << "文件目录：" << argv[1] << std::endl;
    std::vector<boost::filesystem::path> stream = getFiles(in_dirPath, "bin");

    std::vector<std::string> str_vec;
    boost::split(str_vec, in_dirPath, boost::is_any_of("/"), boost::token_compress_on);
    std::string out_dirPath{};
    for (int i = 1; i < str_vec.size() - 2; ++i)
    {
        out_dirPath += "/" + str_vec[i];
    }
    out_dirPath = out_dirPath + "/" + "batch_image/";
    if (!mkDir(out_dirPath))
    {
        std::cout << "创建输出目录失败，请检查。" << std::endl;
        return 0;
    }
    std::cout << "Batch_image_dir path：" << out_dirPath << std::endl;

    for (auto ite = stream.begin(); ite != stream.end(); ++ite)
    {
        std::cout << std::endl;
        std::string out_file = out_dirPath + (*ite).stem().string() + ".bin";

        /*加载点云*/
        std::fstream input((*ite).string().data(), std::ios::in | std::ios::binary);
        if (!input.good())
        {
            std::cerr << "Could not read file: " << (*ite).string() << std::endl;
            exit(EXIT_FAILURE);
        }
        else
        {
            std::cout << ">---process: " << (*ite).string() << std::endl;
        }
        input.seekg(0, std::ios::beg);

        std::vector<PointXYZI> cloud_data;

        int i{};
        for (i = 0; input.good() && !input.eof(); i++)
        {
            PointXYZI point;
            input.read((char *)&point.x, 3 * sizeof(float));
            input.read((char *)&point.intensity, sizeof(float));
            cloud_data.push_back(point);
        }
        input.close();

        pfe_process pfe;
        pfe.process(&cloud_data[0], cloud_data.size());

        /*把pillar化的点云保存为batch_image*/
        std::ofstream out(out_file, std::ios::out | std::ios::binary);
        out.write((char *)pfe.float_buf_.get(), pfe.input_c_ * pfe.input_h_ * pfe.input_w_ * sizeof(float));
        out.close();
        std::cout << std::endl;
    }

    printf("\n=========================");
    printf("\n Preposecc done!");
    printf("\n=========================\n");
}