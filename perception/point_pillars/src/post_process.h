#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include "pp_base.h"
#include <chrono>
struct PostRet
{
    MatrixXf boxes;
    MatrixXf scores;
    MatrixXi labels;

    PostRet(MatrixXf &&boxesi, MatrixXf &&scoresi, MatrixXi &&labelsi)
        : boxes(std::move(boxesi)), scores(std::move(scoresi)), labels(std::move(labelsi)) {}
    PostRet(PostRet &&other) noexcept;
};

void parse_pointpillars_out(float nms_score_threshold, int group_id, int cls_in_shape_0,
                            int boxes_in_shape_0, int dir_cls_in_shape_0, int in_shape_12,
                            int down_ratio,
                            std::vector<float> cls_data,
                            std::vector<float> box_data,
                            std::vector<float> dir_data,
                            std::vector<int> &keep_ids, std::vector<float> &scores,
                            std::vector<int> &cls_argmax, std::vector<int> &dir_cls_argmax,
                            std::vector<std::vector<float>> &boxes);


PostRet
post_process(const vector<MatrixXf> &batch_boxes,
             const vector<MatrixXf> &batch_cls,
             const vector<MatrixXf> &batch_dir_cls,
             const MatrixXf &anchors);

PostRet post_process_1(const vector<vector<float>> &boxes,
                       const vector<float> &scores,
                       const vector<int> &cls,
                       const vector<int> &dir_cls,
                       const vector<int> &keep_ids,
                       const MatrixXf &anchors);

#endif