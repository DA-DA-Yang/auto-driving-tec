#include "post_process.h"

PostRet::PostRet(PostRet &&other) noexcept
{
    // std::cout << "PostRet move copy" << endl;
    boxes = std::move(other.boxes);
    scores = std::move(other.scores);
    labels = std::move(other.labels);
}

MatrixXf decode(const MatrixXf &boxes, const MatrixXf &anchors)
{
    //box:x,y,z,dx,dy,dz,r
    MatrixXf out(boxes.rows(), boxes.cols());
    MatrixXf anchors_z = anchors.col(2) + anchors.col(5) / 2.0;
    // diagonal：斜对角线长度，sqrt(dx^2+dy^2)
    VectorXf diagonal = (anchors.col(3).array().square() + anchors.col(4).array().square()).sqrt();
    //乘以系数
    out.col(0) = boxes.col(0).cwiseProduct(diagonal) + anchors.col(0);
    out.col(1) = boxes.col(1).cwiseProduct(diagonal) + anchors.col(1);
    out.col(2) = boxes.col(2).cwiseProduct(anchors.col(5)) + anchors_z;

    out.col(3) = boxes.col(3).array().exp() * anchors.col(3).array();
    out.col(4) = boxes.col(4).array().exp() * anchors.col(4).array();
    out.col(5) = boxes.col(5).array().exp() * anchors.col(5).array();

    out.col(6) = boxes.col(6) + anchors.col(6);
    // out.col(2) = out.col(2) - out.col(5) / 2.0;
    return out;
}

MatrixXi argmax(const MatrixXf &input, int dim = 1)
{
    if (0 == dim) // 列
    {
        MatrixXi out(1, input.cols());
        Index am;
        for (int i = 0; i < input.cols(); ++i)
        {
            input.col(i).maxCoeff(&am);
            out.col(i) << (am);
        }
        return out;
    }
    else if (1 == dim) // 行
    {
        MatrixXi out(input.rows(), 1);
        Index am;
        for (int i = 0; i < input.rows(); ++i)
        {
            input.row(i).maxCoeff(&am);
            out.row(i) << (am);
        }
        return out;
    }
    else
    {
        throw invalid_argument("dim should be 0 or 1");
    }
}

// (n, 4, 2) -> (n, 4) 4:(xmin, ymin, xmax, ymax)
MatrixXf corner_to_standup_2d(const vector<MatrixXf> &corner)
{
    int num = corner.size();
    MatrixXf standup(num, 4);
    for (int n = 0; n < num; ++n)
    {
        standup.block<1, 2>(n, 0) = corner[n].colwise().minCoeff();
        standup.block<1, 2>(n, 2) = corner[n].colwise().maxCoeff();
    }
    return standup;
}

vector<MatrixXf> center_to_corner_box2d(const MatrixXf &center,
                                        const MatrixXf &dim,
                                        const MatrixXf &rot)
{
    int num = center.rows();
    vector<MatrixXf> corner(num, MatrixXf(4, 2));
    for (int n = 0; n < num; ++n)
    {
        MatrixXf corner_norm(4, 2); // 四个角点逆时针排布：x0y0, x0y1, x1y1, x1y0
        corner_norm.row(0) << dim(n, 0) * (-0.5), dim(n, 1) * (-0.5);
        corner_norm.row(1) << dim(n, 0) * (-0.5), dim(n, 1) * (0.5);
        corner_norm.row(2) << dim(n, 0) * (0.5), dim(n, 1) * (0.5);
        corner_norm.row(3) << dim(n, 0) * (0.5), dim(n, 1) * (-0.5);

        MatrixXf rot_mat(2, 2); // 2D rotation
        rot_mat.row(0) << cos(rot(n, 0)), -sin(rot(n, 0));
        rot_mat.row(1) << sin(rot(n, 0)), cos(rot(n, 0));

        MatrixXf corner_norm_rotated = corner_norm * rot_mat;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                corner_norm_rotated(i, j) += center(n, j);
            }
        }
        corner[n] = corner_norm_rotated;
    }
    return corner;
}

vector<int> sort_vec(const vector<float> &scores, bool ascending = true)
{
    vector<int> idx(scores.size());
    iota(idx.begin(), idx.end(), 0);
    if (ascending)
    {
        auto rule = [&scores](int i, int j) -> bool
        {
            return scores[i] < scores[j];
        };
        stable_sort(idx.begin(), idx.end(), rule); // std::sort默认升序排列
        return idx;
    }
    else // 降序排列
    {
        auto rule = [&scores](int i, int j) -> bool
        {
            return scores[i] > scores[j];
        };
        stable_sort(idx.begin(), idx.end(), rule);
        return idx;
    }
}

float iou(const VectorXf &a, const VectorXf &b)
{
    float left = max(a[0], b[0]);
    float right = min(a[2], b[2]);
    float top = max(a[1], b[1]);
    float bottom = min(a[3], b[3]);
    float width = max((right - left + 1), float(0));
    float height = max((bottom - top + 1), float(0));
    float interS = width * height;
    float Sa = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float Sb = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interS / (Sa + Sb - interS);
}

vector<int> nms(const MatrixXf &boxes_standup, const vector<float> &scores)
{

    vector<int> ind = sort_vec(scores, false);
    int pre_keep = (num_pre_max_size < ind.size()) ? num_pre_max_size : ind.size();
    vector<int> ind_pre_max{ind.begin(), ind.begin() + pre_keep};

    // int num_input = boxes_standup.rows();
    MatrixXf boxes_standup_ordered(pre_keep, boxes_standup.cols());
    for (int i = 0; i < pre_keep; ++i)
    {
        boxes_standup_ordered.row(i) = boxes_standup.row(ind[i]);
    }
    MatrixXf ious(pre_keep, pre_keep);
    for (int i = 0; i < pre_keep; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            ious(i, j) = iou(boxes_standup_ordered.row(i), boxes_standup_ordered.row(j));
        }
    }

    vector<int> keep(pre_keep, 1);
    vector<int> keep_ids;
    for (int i = 0; i < pre_keep; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (1 == keep[j] && ious(i, j) > nms_iou_threshold)
            {
                keep[i] = 0;
                break;
            }
        }
        if (1 == keep[i])
        {
            keep_ids.push_back(ind[i]);
        }
    }

    return keep_ids;
}

PostRet post_process_1(const vector<vector<float>> &in_boxes,
                       const vector<float> &in_scores,
                       const vector<int> &in_cls,
                       const vector<int> &in_dir_cls,
                       const vector<int> &in_keep_ids,
                       const MatrixXf &in_anchors)
{
    //初始box的总数
    int keep_size = in_boxes.size();
    int num_box_features = in_boxes[0].size();
    MatrixXf boxes(keep_size, num_box_features);
    MatrixXf anchors(keep_size, num_box_features);
    MatrixXi cls(keep_size, 1);
    MatrixXi dir_cls(keep_size, 1);
    //拷贝原始值
    for (int i = 0; i < keep_size; ++i)
    {
        cls(i, 0) = in_cls[i];
        dir_cls(i, 0) = in_dir_cls[i];
        //根据解析rpn数据时获得的keep_id，拷贝对应特征的anchor
        anchors.row(i) = in_anchors.row(in_keep_ids[i]);
        for (int j = 0; j < num_box_features; ++j)
        {
            boxes(i, j) = in_boxes[i][j];
        }
    }
    // 根据anchors，给box解码，boxes_decoded：x,y,z,w,l,h,r
    MatrixXf boxes_decoded = decode(boxes, anchors);
    //xy坐标
    MatrixXf center_xy(keep_size, 2);
    for (int i = 0; i < 2; ++i)
    {
        center_xy.col(i) = boxes_decoded.col(i);
    }
    //宽度与长度：width，length
    MatrixXf dim_wl(keep_size, 2);
    for (int i = 0; i < 2; ++i)
    {
        dim_wl.col(i) = boxes_decoded.col(i + 3);
    }
    //角度：rotation angle
    MatrixXf rot(keep_size, 1);
    rot.col(0) = boxes_decoded.col(6);

    vector<MatrixXf> corner = std::move(center_to_corner_box2d(center_xy, dim_wl, rot));
    MatrixXf standup = std::move(corner_to_standup_2d(corner));

    // 极大值抑制
    vector<int> keep_ids = std::move(nms(standup, in_scores));
    int num_nmsed = keep_ids.size();

    MatrixXf boxes_nmsed(num_nmsed, num_box_features);
    MatrixXi labels_nmsed(num_nmsed, 1);
    MatrixXf scores_nmsed(num_nmsed, 1);
    MatrixXi labels_dir_nmsed(num_nmsed, 1);
    for (int i = 0; i < num_nmsed; ++i)
    {
        boxes_nmsed.row(i) = boxes_decoded.row(keep_ids[i]);
        labels_nmsed.row(i) = cls.row(keep_ids[i]);

        scores_nmsed(i, 0) = in_scores[keep_ids[i]];
        labels_dir_nmsed.row(i) = dir_cls.row(keep_ids[i]);
    }
    for (int i = 0; i < num_nmsed; ++i)
    {
        int opp = (boxes_nmsed(i, 6) > 0) ^ bool(labels_dir_nmsed(i, 0));
        boxes_nmsed(i, 6) += opp * M_PI; // if 1 then pi; if 0 then 0
    }

    PostRet post_ret(std::move(boxes_nmsed), std::move(scores_nmsed), std::move(labels_nmsed));
    return post_ret;
}

PostRet post_process(const vector<MatrixXf> &batch_boxes,
                     const vector<MatrixXf> &batch_cls,
                     const vector<MatrixXf> &batch_dir_cls,
                     const MatrixXf &anchors)
{
    MatrixXf pred_cls = batch_cls[0];
    int num_pred = pred_cls.rows();
    int cols_cls = pred_cls.cols();
    int cols_dir_cls = batch_dir_cls[0].cols();
    int cols_box = batch_boxes[0].cols();
    int cols_anchor = anchors.cols();

    MatrixXf pred_cls_sigmoid = pred_cls.unaryExpr([](float x)
                                                   { return 1 / (1 + exp(-x)); }); // sigmoid
    MatrixXf pred_cls_sigmoid_max = pred_cls_sigmoid.rowwise().maxCoeff();

    MatrixXf boxes(num_pred, cols_box);
    MatrixXf dir_cls(num_pred, cols_dir_cls);
    MatrixXf cls(num_pred, cols_cls);
    vector<float> cls_sigmoid_max(num_pred);
    MatrixXf anchors_kept(num_pred, cols_anchor);

    int num_keep = 0;
    for (int i = 0; i < num_pred; ++i)
    {
        if (pred_cls_sigmoid_max(i, 0) >= nms_score_threshold)
        {
            boxes.row(num_keep) = batch_boxes[0].row(i);
            dir_cls.row(num_keep) = batch_dir_cls[0].row(i);
            cls.row(num_keep) = pred_cls.row(i);
            cls_sigmoid_max[num_keep] = pred_cls_sigmoid_max(i, 0);
            anchors_kept.row(num_keep) = anchors.row(i);
            num_keep += 1;
        }
    }
    // std::cout << "pred_cls_sigmoid_max:" << pred_cls_sigmoid_max << std::endl;
    boxes.conservativeResize(num_keep, cols_box);
    dir_cls.conservativeResize(num_keep, cols_dir_cls);
    cls.conservativeResize(num_keep, cols_cls);
    for (int i = num_keep; i < num_pred; ++i)
    {
        cls_sigmoid_max.pop_back();
    }
    anchors_kept.conservativeResize(num_keep, cols_anchor);

    // std::cout << "num_keep:" << num_keep << std::endl;

    MatrixXf boxes_decoded = decode(boxes, anchors_kept);
    // std::cout << "boxes_decoded.rows():" << boxes_decoded.rows() << std::endl;
    // std::cout << "boxes_decoded.cols():" << boxes_decoded.cols() << std::endl;
    MatrixXi dir_cls_argmax = std::move(argmax(dir_cls));
    //  std::cout << "dir_cls_argmax.rows():" << dir_cls_argmax.rows() << std::endl;
    // std::cout << "dir_cls_argmax.cols():" << dir_cls_argmax.cols() << std::endl;
    MatrixXi cls_sigmoid_argmax = std::move(argmax(cls));
    //  std::cout << "cls_sigmoid_argmax.rows():" << cls_sigmoid_argmax.rows() << std::endl;
    // std::cout << "cls_sigmoid_argmax.cols():" << cls_sigmoid_argmax.cols() << std::endl;

    // boxes_decoded: (num_keep, 7)
    // 7:(x,y,z,w,l,h,rot)
    MatrixXf center_xy(num_keep, 2);
    for (int i = 0; i < 2; ++i)
    {
        center_xy.col(i) = boxes_decoded.col(i);
    }
    MatrixXf dim_wl(num_keep, 2);
    for (int i = 0; i < 2; ++i)
    {
        dim_wl.col(i) = boxes_decoded.col(i + 3);
    }
    MatrixXf rot(num_keep, 1);
    rot.col(0) = boxes_decoded.col(6);

    vector<MatrixXf> corner = std::move(center_to_corner_box2d(center_xy, dim_wl, rot));

    MatrixXf standup = std::move(corner_to_standup_2d(corner));

    vector<int> keep_ids = std::move(nms(standup, cls_sigmoid_max));
    int num_nmsed = keep_ids.size();
    // for(auto array: keep_ids) {
    //     std::cout << array << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "num_nmsed:" << num_nmsed << std::endl;
    MatrixXf boxes_nmsed(num_nmsed, cols_box);
    MatrixXi labels_nmsed(num_nmsed, 1);
    MatrixXf scores_nmsed(num_nmsed, 1);
    MatrixXi labels_dir_nmsed(num_nmsed, 1);
    for (int i = 0; i < num_nmsed; ++i)
    {
        boxes_nmsed.row(i) = boxes_decoded.row(keep_ids[i]);
        labels_nmsed.row(i) = cls_sigmoid_argmax.row(keep_ids[i]);

        scores_nmsed(i, 0) = cls_sigmoid_max[keep_ids[i]];
        labels_dir_nmsed.row(i) = dir_cls_argmax.row(keep_ids[i]);
    }
    for (int i = 0; i < num_nmsed; ++i)
    {
        int opp = (boxes_nmsed(i, 6) > 0) ^ bool(labels_dir_nmsed(i, 0));
        boxes_nmsed(i, 6) += opp * M_PI; // if 1 then pi; if 0 then 0
    }

    // std::cout << "boxes_nmsed.rows():" << boxes_nmsed.rows() << std::endl;
    // std::cout << "boxes_nmsed.cols():" << boxes_nmsed.cols() << std::endl;
    // std::cout << "scores_nmsed.rows():" << scores_nmsed.rows() << std::endl;
    // std::cout << "scores_nmsed.cols():" << scores_nmsed.cols() << std::endl;
    // std::cout << "labels_nmsed.rows():" << labels_nmsed.rows() << std::endl;
    // std::cout << "labels_nmsed.cols():" << labels_nmsed.cols() << std::endl;

    PostRet post_ret(std::move(boxes_nmsed), std::move(scores_nmsed), std::move(labels_nmsed));

    return post_ret;
}

void parse_pointpillars_out(float nms_score_threshold, int group_id, int cls_in_shape_0,
                            int boxes_in_shape_0, int dir_cls_in_shape_0, int in_shape_12,
                            int down_ratio,
                            std::vector<float> cls_data,
                            std::vector<float> box_data,
                            std::vector<float> dir_data,
                            std::vector<int> &keep_ids, std::vector<float> &scores,
                            std::vector<int> &cls_argmax, std::vector<int> &dir_cls_argmax,
                            std::vector<std::vector<float>> &boxes)
{
    std::cout << std::endl;
    int nCount{};
    int cls_out_shape_1 = cls_in_shape_0 / down_ratio;
    int boxes_out_shape_1 = boxes_in_shape_0 / down_ratio;
    int base_id = group_id * in_shape_12 * down_ratio;

    for (int i = 0; i < in_shape_12; ++i)
    {
        for (int j = 0; j < down_ratio; j++)
        {
            int arg_max_cls = -1;
            float max_cls = 0;
            float nms_score_threshold_cur = nms_score_threshold;
            //确定分类得分的最大值，还得和nms阈值比较
            for (int k = 0; k < cls_out_shape_1; ++k)
            {
                int id = i + (j * cls_out_shape_1 + k) * in_shape_12;
                float cur_cls = cls_data[id];
                if (nCount < 11)
                {
                    std::cout << cur_cls << " ";
                    nCount++;
                }
                // 不要计算sigmoid，性能消耗很大！
                // float cur_cls_sigmoid = 1 / (1 + exp(-cur_cls));
                // float cur_cls_sigmoid = cur_cls;
                if (cur_cls >= nms_score_threshold_cur)
                {
                    max_cls = cur_cls;
                    arg_max_cls = k;
                    nms_score_threshold_cur = cur_cls;
                }
            }
            //如果当前分类值得信任，则保存box的参数与dir的参数
            if (arg_max_cls >= 0)
            {
                keep_ids.push_back(base_id + i * down_ratio + j);
                scores.push_back(max_cls);
                cls_argmax.push_back(arg_max_cls);
                //保存box的参数
                std::vector<float> cur_box(boxes_out_shape_1);
                for (int k = 0; k < boxes_out_shape_1; ++k)
                {
                    int id = i + (j * boxes_out_shape_1 + k) * in_shape_12;
                    float cur_boxes = box_data[id];
                    cur_box[k] = cur_boxes;
                }
                boxes.push_back(cur_box);

                int id = i + (j * 2 + 0) * in_shape_12;
                float dir_cls_0 = dir_data[id];
                int id1 = i + (j * 2 + 1) * in_shape_12;
                float dir_cls_1 = dir_data[id1];
                int cur_dir_cls_argmax = (dir_cls_0 > dir_cls_1) ? 0 : 1;
                dir_cls_argmax.push_back(cur_dir_cls_argmax);
            }
        }
    }
}