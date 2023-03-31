#include "autoware_pointpillar.h"
#include "pfe_process.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    // 该程序用于执行点云pillar化。
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少文件路径！" << std::endl;
        return 0;
    }
    std::cout << "输入文件：" << argv[1] << std::endl;
    std::string file_path(argv[1]);

    // 结果输出路径
    std::string res_dir_path = "/home/yangda/my_project/perception_lidar/tmp/temp/";

    /*加载点云*/

    std::fstream input(file_path.data(), std::ios::in | std::ios::binary);
    if (!input.good())
    {
        std::cerr << "Could not read file: " << file_path << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);

    std::vector<float> point_array;

    int i{};
    for (i = 0; input.good() && !input.eof(); i++)
    {
        float t;
        input.read((char *)&t, 1 * sizeof(float));
        point_array.push_back(t);
    }
    input.close();
    std::cout << "pointcloud size: " << point_array.size() / 4 << std::endl;

    Autoware_PointPillar pfe;
    pfe.preProcess(&point_array[0], point_array.size() / 4);

    /*把pillar化的点云保存为batch_image*/
    std::ofstream out;
    out.open(res_dir_path + "pillar_x.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_pillar_x_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "pillar_y.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_pillar_y_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "pillar_z.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_pillar_z_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "pillar_i.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_pillar_i_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "x_coors_for_sub_shaped.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_x_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "y_coors_for_sub_shaped.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_y_coors_for_sub_shaped_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "pillar_feature_mask.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_pillar_feature_mask_, MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "num_points_per_pillar.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_num_points_per_pillar_, MAX_NUM_PILLARS_ * sizeof(float));
    out.close();

    out.open(res_dir_path + "x_coors.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_x_coors_, MAX_NUM_PILLARS_ * sizeof(int));
    out.close();

    out.open(res_dir_path + "y_coors.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_y_coors_, MAX_NUM_PILLARS_ * sizeof(int));
    out.close();

    out.open(res_dir_path + "sparse_pillar_map.bin", std::ios::out | std::ios::binary);
    out.write((char *)pfe.dev_sparse_pillar_map_, GRID_X_SIZE_ * GRID_Y_SIZE_ * sizeof(int));
    out.close();

    printf("pillar count:%i", pfe.host_pillar_count_[0]);
    printf("\n=========================");
    printf("\nDone!");
    printf("\n=========================\n");
}