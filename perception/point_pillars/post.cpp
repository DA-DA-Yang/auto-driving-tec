#include "post_process.h"
#include "anchor.h"
#include "base.h"

int main(int argc, char **argv)
{
    printf("\n");
    printf("\n>>>----------post process----------<<<\n");
    if (argc <= 1)
    {
        std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
        return 0;
    }
    
    std::string in_dirPath(argv[1]);
    std::cout << "文件目录：" << argv[1] << std::endl;
    
    // 获取上级文件目录
    std::vector<std::string> str_vec;
    boost::split(str_vec, in_dirPath, boost::is_any_of("/"), boost::token_compress_on);
    std::string up_dirPath{};
    for (int i = 1; i < str_vec.size() - 2; ++i)
    {
        up_dirPath += "/" + str_vec[i];
    }

    in_dirPath = up_dirPath + "/" + "rpn_data/";
    if (access(in_dirPath.c_str(), F_OK) == -1)
    {
        std::cout << "读取batch_image目录失败，请检查。" << std::endl;
        return 0;
    }
    std::cout << "Rpn_data_dir: " << in_dirPath << std::endl;
    // 获取文件目录
    std::vector<boost::filesystem::path> stream = getFiles(in_dirPath, "bin");

    std::string out_dirPath = up_dirPath + "/" + "boxes/";
    if (!mkDir(out_dirPath))
    {
        std::cout << "创建输出目录失败，请检查。" << std::endl;
        return 0;
    }
    std::cout << "Boxes_dir: " << out_dirPath << std::endl;

    for (auto ite = stream.begin(); ite != stream.end(); ++ite)
    {
        std::cout << std::endl;
        std::string file_path = (*ite).string();
        std::string out_path_txt = out_dirPath + (*ite).stem().string() + ".txt";
        if(access(file_path.c_str(),F_OK)==-1)
        {
            std::cout << "Read file failed: " << file_path << std::endl;
        }
        else
        {
            std::cout << ">---process: " << file_path << std::endl;
        }
        // 读取推理结果
        std::fstream input(file_path.data(), std::ios::in | std::ios::binary);
        if (!input.good())
        {
            std::cerr << "Could not read file: " << file_path << std::endl;
            exit(EXIT_FAILURE);
        }
        input.seekg(0, std::ios::beg);
        std::vector<std::vector<float>> rpn_data;
        std::vector<std::vector<int>> rpn_data_size;
        int total{};
        for (int i = 0; i < 4; i++)
        {
            // dayang: a1000-pointPillar.onnx
            std::vector<int> cls_size{1, 8, 200, 176};
            std::vector<int> box_size{1, 14, 200, 176};
            std::vector<int> dir_size{1, 4, 200, 176};
            rpn_data_size.push_back(cls_size);
            rpn_data_size.push_back(box_size);
            rpn_data_size.push_back(dir_size);
            std::vector<float> cls_data;
            cls_data.resize(cls_size[0] * cls_size[1] * cls_size[2] * cls_size[3]);
            rpn_data.push_back(cls_data);
            std::vector<float> box_data;
            box_data.resize(box_size[0] * box_size[1] * box_size[2] * box_size[3]);
            rpn_data.push_back(box_data);
            std::vector<float> dir_data;
            dir_data.resize(dir_size[0] * dir_size[1] * dir_size[2] * dir_size[3]);
            rpn_data.push_back(dir_data);
            total += cls_data.size() + box_data.size() + dir_data.size();
        }
        std::vector<float> tmp_data;
        for (int i = 0; input.good() && !input.eof(); i++)
        {
            if (i >= rpn_data_size.size())
            {
                float tmp;
                input.read((char *)&tmp, 1 * sizeof(float));
                // 这里要加限制，否则就会出现多读一个数据的情况
                if (!input.eof())
                {
                    std::cout << "数据尚未读取完，可能是存入出错，请检查！" << std::endl;
                    return 0;
                }
                else
                {
                    std::cout << "数据读取成功！" << std::endl;
                    break;
                }
            }
            // 读取数据
            int total_size = rpn_data_size[i][0] * rpn_data_size[i][1] * rpn_data_size[i][2] * rpn_data_size[i][3];
            input.read((char *)&rpn_data[i][0], total_size * sizeof(float));
        }
        input.close();
        // 检查首尾数据
        auto tmp_value = *((*(rpn_data.begin())).begin());
        std::cout << "first value: " << tmp_value << std::endl;
        tmp_value = rpn_data.back().back();
        std::cout << "last value: " << tmp_value << std::endl;

        // 解析rpn推理结果
        std::vector<int> keep_ids;
        std::vector<float> scores;
        std::vector<int> cls_argmax;
        std::vector<int> dir_cls_argmax;
        std::vector<std::vector<float>> boxes;
        for (int i = 0; i < 4; i++)
        {
            int cls_index = i * 3;
            int box_index = i * 3 + 1;
            int dir_index = i * 3 + 2;
            parse_pointpillars_out(nms_score_threshold, i,
                                   rpn_data_size[cls_index][1], rpn_data_size[box_index][1], rpn_data_size[dir_index][1],
                                   rpn_data_size[cls_index][2] * rpn_data_size[cls_index][3], 2,
                                   rpn_data[cls_index], rpn_data[box_index], rpn_data[dir_index],
                                   keep_ids, scores, cls_argmax, dir_cls_argmax, boxes);
        }
        // 打印rpn推理相关结果
        std::cout << "\n----rpn inference: ----" << std::endl
                  << "keep id size: " << keep_ids.size() << std::endl
                  << "scores size: " << scores.size() << std::endl
                  << "cls argmax size: " << cls_argmax.size() << std::endl
                  << "dir cls argmax size: " << dir_cls_argmax.size() << std::endl
                  << "boxes size: " << boxes.size() << std::endl;

        // 后处理
        MatrixXf anchors = std::move(create_anchor());
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
    }
    std::cout << "=======================" << std::endl
              << "Post process done!" << std::endl
              << "=======================" << std::endl;

}