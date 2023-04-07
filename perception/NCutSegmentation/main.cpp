
#include "ncut.h"
#include "ncut_segmentation.h"
#include "base.h"

int main(int argc, char **argv)
{
    printf("\n");
    printf("\n>>>----------NCut segmentation----------<<<\n");
    // 该程序用于执行点云pillar化。
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
        return 0;
    }
    // 获取文件目录
    std::string in_dirPath(argv[1]);
    std::cout << "文件目录：" << argv[1] << std::endl;
    std::vector<boost::filesystem::path> stream = getFiles(in_dirPath, "bin");

    // 对每帧点云进行处理
    for (auto ite = stream.begin(); ite != stream.end(); ++ite)
    {
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
            input.read((char *)&point.i, sizeof(float));
            cloud_data.push_back(point);
        }
        input.close();

        NCutSegmentation ncut_seg;
        ncut_seg.getSegments(cloud_data);
    }

    return 0;
}