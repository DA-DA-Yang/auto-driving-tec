#include "post_process.h"
#include "autoware_pointpillar.h"

int main(int argc, char **argv)
{
    //需要输入一个参数：out_file_name，如：0001
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少输出文件名！" << std::endl;
        return 0;
    }
    std::string output_file_name(argv[1]);
    //输入数据目录
    std::string data_dir_path = "/home/yangda/my_project/perception_lidar/tmp/temp/";
    //输出数据目录
    std::string res_dir_path = "/home/yangda/my_project/perception_lidar/tmp/";
    std::string rpn_data_path = data_dir_path + "rpn_data.bin";
    std::string sparse_pillar_map_path = data_dir_path + "sparse_pillar_map.bin";
    std::string out_path_txt = res_dir_path + output_file_name + ".txt";

    // 读取推理结果
    std::fstream input_rpn(rpn_data_path.data(), std::ios::in | std::ios::binary);
    if (!input_rpn.good())
    {
        std::cerr << "Could not read file: " << rpn_data_path << std::endl;
        exit(EXIT_FAILURE);
    }
    input_rpn.seekg(0, std::ios::beg);
    std::vector<std::vector<float>> rpn_data;
    std::vector<std::vector<int>> rpn_data_size;
    int total{};
    for (int i = 0; i < 1; i++)
    {
        // dayang: kitti-rpn.onnx
        std::vector<int> box_size{1, 14, 248, 216};
        std::vector<int> cls_size{1, 2, 248, 216};
        std::vector<int> dir_size{1, 4, 248, 216};
        rpn_data_size.push_back(box_size);
        rpn_data_size.push_back(cls_size);
        rpn_data_size.push_back(dir_size);

        std::vector<float> box_data;
        box_data.resize(box_size[0] * box_size[1] * box_size[2] * box_size[3]);
        rpn_data.push_back(box_data);
        std::vector<float> cls_data;
        cls_data.resize(cls_size[0] * cls_size[1] * cls_size[2] * cls_size[3]);
        rpn_data.push_back(cls_data);
        std::vector<float> dir_data;
        dir_data.resize(dir_size[0] * dir_size[1] * dir_size[2] * dir_size[3]);
        rpn_data.push_back(dir_data);

        total += cls_data.size() + box_data.size() + dir_data.size();
    }
    for (int i = 0; input_rpn.good() && !input_rpn.eof(); i++)
    {
        if (i >= rpn_data_size.size())
        {
            // 多读一个数据的情况，如果为文件结尾，表明数据全部读取，否则表明数据未读完。
            float tmp;
            input_rpn.read((char *)&tmp, 1 * sizeof(float));
            if (!input_rpn.eof())
            {
                std::cout << "rpn数据尚未读取完，可能是存入出错，请检查！" << std::endl;
                return 0;
            }
            else
            {
                std::cout << "rpn数据读取成功！" << std::endl;
                break;
            }
        }
        // 读取数据
        int total_size = rpn_data_size[i][0] * rpn_data_size[i][1] * rpn_data_size[i][2] * rpn_data_size[i][3];
        input_rpn.read((char *)&rpn_data[i][0], total_size * sizeof(float));
    }
    input_rpn.close();
    // 检查首尾数据
    auto tmp_value = *((*(rpn_data.begin())).begin());
    std::cout << "first value: " << tmp_value << std::endl;
    tmp_value = rpn_data.back().back();
    std::cout << "last value: " << tmp_value << std::endl;

    // 读取sparse_pillar_map，用于计算anchor_mask
    std::vector<int> sparse_pillar_map;
    sparse_pillar_map.resize(GRID_X_SIZE_ * GRID_Y_SIZE_);
    std::fstream input_pillarmap(sparse_pillar_map_path.data(), std::ios::in | std::ios::binary);
    if (!input_pillarmap.good())
    {
        std::cerr << "Could not read file: " << sparse_pillar_map_path << std::endl;
        exit(EXIT_FAILURE);
    }
    input_pillarmap.seekg(0, std::ios::beg);
    {
        // 读取数据
        input_pillarmap.read((char *)&sparse_pillar_map[0], GRID_X_SIZE_ * GRID_Y_SIZE_ * sizeof(int));
        //再多读一个，检查是否读取完毕
        int tmp;
        input_pillarmap.read((char *)&tmp, 1 * sizeof(int));
        if (!input_pillarmap.eof())
        {
            std::cout << "sparse_pillar_map数据尚未读取完，可能是存入出错，请检查！" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "sparse_pillar_map数据读取成功！" << std::endl;
        }
    }
    input_pillarmap.close();

    // ---解析rpn推理结果---
    // 计算anchor
    Autoware_Anchor aa;
    MatrixXf anchors = std::move(aa.create_anchor());
    // 计算anchor_mask
    std::vector<int> anchor_mask = std::move(aa.getAnchorMask(&sparse_pillar_map[0]));
    std::vector<int> keep_ids;
    std::vector<float> scores;
    std::vector<int> cls_argmax;
    std::vector<int> dir_cls_argmax;
    std::vector<std::vector<float>> boxes;
    for (int i = 0; i < 1; i++)
    {
        
        int box_index = i * 3;
        int cls_index = i * 3 + 1;
        int dir_index = i * 3 + 2;
        //解析数据
        parse_rpn_out(nms_score_threshold,
                      rpn_data_size[cls_index][1] / 2, rpn_data_size[box_index][1] / 2, rpn_data_size[dir_index][1] / 2,
                      rpn_data_size[cls_index][2] * rpn_data_size[cls_index][3] * 2,
                      rpn_data[cls_index], rpn_data[box_index], rpn_data[dir_index], anchor_mask, keep_ids, scores, cls_argmax, dir_cls_argmax, boxes);
    }
    // 打印rpn推理相关结果
    std::cout << "----rpn inference: ----" << std::endl
              << "keep id size: " << keep_ids.size() << std::endl
              << "scores size: " << scores.size() << std::endl
              << "cls argmax size: " << cls_argmax.size() << std::endl
              << "dir cls argmax size: " << dir_cls_argmax.size() << std::endl
              << "boxes size: " << boxes.size() << std::endl;

    // 后处理
    auto post_t0 = std::chrono::high_resolution_clock::now();
    PostRet post_ret = std::move(post_process_1(boxes, scores, cls_argmax, dir_cls_argmax, keep_ids, anchors));
    auto post_t1 = std::chrono::high_resolution_clock::now();
    auto post_cost = std::chrono::duration_cast<std::chrono::microseconds>(post_t1 - post_t0).count();
    // 打印感知结果
    cout << "output Rst:" << endl;
    std::cout << "Postprocess time used: " << post_cost / std::pow(10, 3) << " ms" << std::endl;
    std::cout << "post_ret.boxes.rows():" << post_ret.boxes.rows() << std::endl
              << "post_ret.boxes.cols():" << post_ret.boxes.cols() << std::endl
              << "post_ret.scores.rows():" << post_ret.scores.rows() << std::endl
              << "post_ret.scores.cols():" << post_ret.scores.cols() << std::endl
              << "post_ret.labels.rows():" << post_ret.labels.rows() << std::endl
              << "post_ret.labels.cols():" << post_ret.labels.cols() << std::endl;
    std::cout << post_ret.boxes << std::endl
              << std::endl
              << post_ret.scores << std::endl
              << std::endl
              << post_ret.labels << std::endl
              << std::endl;

    // 输出感知结果，用于3d显示
    std::vector<float> out_detections;
    std::vector<int> out_labels;
    out_detections.clear();
    out_labels.clear();
    for (int i = 0; i < post_ret.boxes.rows(); ++i)
    {
        for (int idx = 0; idx < 7; ++idx)
        {
            out_detections.push_back(post_ret.boxes(i, idx));
        }
        out_labels.push_back(post_ret.labels(i, 0));
    }

    std::ofstream txtfileOut(out_path_txt, std::ios::out);
    for (size_t idx = 0, idx_max = out_labels.size(); idx < idx_max; idx++)
    {
        for (int i = 0; i < 7; i++)
            txtfileOut << out_detections[i + idx * 7] << " ";
        txtfileOut << out_labels[idx] << std::endl;
    }
    txtfileOut.close();

    std::cout << "=======================" << std::endl
              << "Done!" << std::endl
              << "=======================" << std::endl;
}