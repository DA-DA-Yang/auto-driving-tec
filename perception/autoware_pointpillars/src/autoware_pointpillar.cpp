#include "autoware_pointpillar.h"
#include <cmath>

Autoware_PointPillar::Autoware_PointPillar(/* args */)
{
    dev_pillar_x_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_pillar_y_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_pillar_z_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_pillar_i_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_num_points_per_pillar_ = new float[MAX_NUM_PILLARS_]{0};
    dev_x_coors_for_sub_shaped_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_y_coors_for_sub_shaped_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};
    dev_pillar_feature_mask_ = new float[MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_]{0};

    dev_x_coors_ = new int[MAX_NUM_PILLARS_]{0};
    dev_y_coors_ = new int[MAX_NUM_PILLARS_]{0};

    dev_sparse_pillar_map_ = new int[GRID_Y_SIZE_ * GRID_X_SIZE_]{0};
}

Autoware_PointPillar::~Autoware_PointPillar()
{
}

void Autoware_PointPillar::preProcess(const float *in_points_array, const int in_num_points)
{

    pre_process(in_points_array, in_num_points, dev_sparse_pillar_map_, dev_x_coors_, dev_y_coors_, dev_num_points_per_pillar_, dev_pillar_x_,
                dev_pillar_y_, dev_pillar_z_, dev_pillar_i_, dev_x_coors_for_sub_shaped_, dev_y_coors_for_sub_shaped_,
                dev_pillar_feature_mask_, host_pillar_count_);
}

void Autoware_PointPillar::pre_process(const float *in_points_array, int in_num_points, int *sparse_pillar_map, int *x_coors, int *y_coors,
                                       float *num_points_per_pillar, float *pillar_x, float *pillar_y, float *pillar_z,
                                       float *pillar_i, float *x_coors_for_sub_shaped, float *y_coors_for_sub_shaped,
                                       float *pillar_feature_mask, int *host_pillar_count)
{
    int pillar_count = 0;
    float x_coors_for_sub[MAX_NUM_PILLARS_] = {0};
    float y_coors_for_sub[MAX_NUM_PILLARS_] = {0};
    // init variables
    int coor_to_pillaridx[GRID_Y_SIZE_ * GRID_X_SIZE_];
    // 这里会把coor_to_pillaridx全赋值为-1
    initializeVariables(coor_to_pillaridx, pillar_x, pillar_y, pillar_z, pillar_i,
                        x_coors_for_sub_shaped, y_coors_for_sub_shaped);
    //遍历点云的每个point
    for (int i = 0; i < in_num_points; i++)
    {
        //计算当前point所在pillar的坐标，相当于把点云网格化
        //dayang:这里点云范围被限制在预先设定的范围中
        int x_coor = std::floor((in_points_array[i * NUM_BOX_CORNERS_ + 0] - MIN_X_RANGE_) / PILLAR_X_SIZE_);
        int y_coor = std::floor((in_points_array[i * NUM_BOX_CORNERS_ + 1] - MIN_Y_RANGE_) / PILLAR_Y_SIZE_);
        int z_coor = std::floor((in_points_array[i * NUM_BOX_CORNERS_ + 2] - MIN_Z_RANGE_) / PILLAR_Z_SIZE_);
        if (x_coor < 0 || x_coor >= GRID_X_SIZE_ || y_coor < 0 || y_coor >= GRID_Y_SIZE_ || z_coor < 0 ||
            z_coor >= GRID_Z_SIZE_)
        {
            continue;
        }
        // reverse index
        int pillar_index = coor_to_pillaridx[y_coor * GRID_X_SIZE_ + x_coor];
        //=-1说明尚未进行pillar操作
        if (pillar_index == -1)
        {
            //pillar的索引
            pillar_index = pillar_count;
            //如果超过预先设定的最大数目，则不再进行pillar化
            if (pillar_count >= MAX_NUM_PILLARS_)
            {
                break;
            }
            //pillar数增加
            pillar_count += 1;
            coor_to_pillaridx[y_coor * GRID_X_SIZE_ + x_coor] = pillar_index;
            //pillar的坐标索引，表示在第几个网格
            y_coors[pillar_index] = std::floor(y_coor);
            x_coors[pillar_index] = std::floor(x_coor);

            //这里是计算实际的坐标值
            y_coors_for_sub[pillar_index] = std::floor(y_coor) * PILLAR_Y_SIZE_ + MIN_Y_RANGE_; // 原为-39.9f
            x_coors_for_sub[pillar_index] = std::floor(x_coor) * PILLAR_X_SIZE_ + MIN_X_RANGE_; // 原为0.1f

            sparse_pillar_map[y_coor * GRID_X_SIZE_ + x_coor] = 1;
        }
        //确定当前pillar中的点数
        int num = num_points_per_pillar[pillar_index];
        //如果小于100，则继续增加
        if (num < MAX_NUM_POINTS_PER_PILLAR_)
        {
            pillar_x[pillar_index * MAX_NUM_POINTS_PER_PILLAR_ + num] = in_points_array[i * NUM_BOX_CORNERS_ + 0];
            pillar_y[pillar_index * MAX_NUM_POINTS_PER_PILLAR_ + num] = in_points_array[i * NUM_BOX_CORNERS_ + 1];
            pillar_z[pillar_index * MAX_NUM_POINTS_PER_PILLAR_ + num] = in_points_array[i * NUM_BOX_CORNERS_ + 2];
            pillar_i[pillar_index * MAX_NUM_POINTS_PER_PILLAR_ + num] = in_points_array[i * NUM_BOX_CORNERS_ + 3];
            num_points_per_pillar[pillar_index] += 1;
        }
    }

    for (int i = 0; i < MAX_NUM_PILLARS_; i++)
    {
        //x,y为pillar的真实坐标
        float x = x_coors_for_sub[i];
        float y = y_coors_for_sub[i];
        //当前pillar包含的点数
        int num_points_for_a_pillar = num_points_per_pillar[i];
        for (int j = 0; j < MAX_NUM_POINTS_PER_PILLAR_; j++)
        {
            // 为pillar中每个point赋值pillar的真实坐标
            x_coors_for_sub_shaped[i * MAX_NUM_POINTS_PER_PILLAR_ + j] = x;
            y_coors_for_sub_shaped[i * MAX_NUM_POINTS_PER_PILLAR_ + j] = y;
            // 如果j < num_points_for_a_pillar)表明为有效点，反之，为0
            if (j < num_points_for_a_pillar)
            {
                pillar_feature_mask[i * MAX_NUM_POINTS_PER_PILLAR_ + j] = 1.0f;
            }
            else
            {
                pillar_feature_mask[i * MAX_NUM_POINTS_PER_PILLAR_ + j] = 0.0f;
            }
        }
    }
    host_pillar_count[0] = pillar_count;
}

void Autoware_PointPillar::initializeVariables(int *coor_to_pillaridx, float *pillar_x,
                                           float *pillar_y, float *pillar_z, float *pillar_i,
                                           float *x_coors_for_sub_shaped, float *y_coors_for_sub_shaped)
{
    for (int i = 0; i < GRID_Y_SIZE_; i++)
    {
        for (int j = 0; j < GRID_X_SIZE_; j++)
        {
            coor_to_pillaridx[i * GRID_X_SIZE_ + j] = -1;
        }
    }

    for (int i = 0; i < MAX_NUM_PILLARS_ * MAX_NUM_POINTS_PER_PILLAR_; i++)
    {
        pillar_x[i] = 0;
        pillar_y[i] = 0;
        pillar_z[i] = 0;
        pillar_i[i] = 0;
        x_coors_for_sub_shaped[i] = 0;
        y_coors_for_sub_shaped[i] = 0;
    }
}

void Autoware_PointPillar::postProcess()
{

}

Autoware_Anchor::Autoware_Anchor(/* args */)
{
    Array3i xyz_range;
    for (int i = 0; i < 2; ++i)
    {
        xyz_range[i] = floor((pc_range_high_[i] - pc_range_low_[i]) / stride_[i]);
    }
    xyz_range[2] = 1; 
    int total_count = xyz_range[0] * xyz_range[1] * xyz_range[2] * 2;
    _anchors = MatrixXf(total_count, 7);

    for (int y = 0; y < xyz_range[1]; ++y)
    {
        for (int x = 0; x < xyz_range[0]; ++x)
        {
            int cur_id = 2 * (x + y * xyz_range[0]);

            _anchors.row(cur_id) << offset_[0] + x * stride_[0],
                offset_[1] + y * stride_[1],
                offset_[2] + 1 * stride_[2],
                size_[0],
                size_[1],
                size_[2],
                0;//angle=0

            _anchors.row(cur_id + 1) << offset_[0] + x * stride_[0],
                offset_[1] + y * stride_[1],
                offset_[2] + 1 * stride_[2],
                size_[0],
                size_[1],
                size_[2],
                1.57; // angle=90, M_PI / 2;
        }
    }
}

Autoware_Anchor::~Autoware_Anchor()
{
}

MatrixXf Autoware_Anchor::create_anchor()
{
    int num_anchors = _anchors.rows();
    MatrixXf anchors(num_anchors, box_coder_size);
    int start_row = 0;
    anchors.block(start_row, 0, _anchors.rows(), box_coder_size) = std::move(_anchors);
    return anchors;
}

std::vector<int> Autoware_Anchor::getAnchorMask(int *sparse_pillar_map)
{
    //min_x,min_y,max_x,max_y
    _box_anchors = MatrixXf(NUM_ANCHOR_, 4);
    _anchor_mask.resize(NUM_ANCHOR_);
    // flipping box's dimension
    float flipped_anchors_dx[NUM_ANCHOR_] = {0};
    float flipped_anchors_dy[NUM_ANCHOR_] = {0};
    for (size_t y = 0; y < NUM_ANCHOR_Y_INDS_; y++)
    {
        for (size_t x = 0; x < NUM_ANCHOR_X_INDS_; x++)
        {
            int base_ind = y * NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_R_INDS_ + x * NUM_ANCHOR_R_INDS_;
            flipped_anchors_dx[base_ind + 0] = ANCHOR_DX_SIZE_;
            flipped_anchors_dy[base_ind + 0] = ANCHOR_DY_SIZE_;
            //在旋转角度为90度时，需要反转
            flipped_anchors_dx[base_ind + 1] = ANCHOR_DY_SIZE_;
            flipped_anchors_dy[base_ind + 1] = ANCHOR_DX_SIZE_;
        }
    }
    for (size_t y = 0; y < NUM_ANCHOR_Y_INDS_; y++)
    {
        for (size_t x = 0; x < NUM_ANCHOR_X_INDS_; x++)
        {
            for (size_t r = 0; r < NUM_ANCHOR_R_INDS_; r++)
            {
                int ind = y * NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_R_INDS_ + x * NUM_ANCHOR_R_INDS_ + r;
                // 计算box的四个角点的xy坐标：
                //_anchors(ind, 0)=px，_anchors(ind, 1)=py，anchor的xy坐标
                // px-ANCHOR_DX_SIZE_/2，left-x
                // py-ANCHOR_DY_SIZE_/2，left-y
                // px+ANCHOR_DX_SIZE_/2, right-x
                // py+ANCHOR_DY_SIZE_/2, right-y
                _box_anchors(ind, 0) = _anchors(ind, 0) - flipped_anchors_dx[ind] / 2.0f;
                _box_anchors(ind, 1) = _anchors(ind, 1) - flipped_anchors_dy[ind] / 2.0f;
                _box_anchors(ind, 2) = _anchors(ind, 0) + flipped_anchors_dx[ind] / 2.0f;
                _box_anchors(ind, 3) = _anchors(ind, 1) + flipped_anchors_dy[ind] / 2.0f;

                int anchor_coor[4] = {0};
                //left-x
                anchor_coor[0] =
                    floor((_box_anchors(ind, 0) - pc_range_low(0)) / PILLAR_X_SIZE_);
                //bottom-y
                anchor_coor[1] =
                    floor((_box_anchors(ind, 1) - pc_range_low(1)) / PILLAR_Y_SIZE_);
                //right-x
                anchor_coor[2] =
                    floor((_box_anchors(ind, 2) - pc_range_low(0)) / PILLAR_X_SIZE_);
                //top-y
                anchor_coor[3] =
                    floor((_box_anchors(ind, 3) - pc_range_low(1)) / PILLAR_Y_SIZE_);
                anchor_coor[0] = std::max(anchor_coor[0], 0);
                anchor_coor[1] = std::max(anchor_coor[1], 0);
                anchor_coor[2] = std::min(anchor_coor[2], GRID_X_SIZE_);
                anchor_coor[3] = std::min(anchor_coor[3], GRID_Y_SIZE_);
                //2---1
                //3---4
                for (int j = anchor_coor[1]; j < anchor_coor[3]; ++j)
                {
                    int res = 0;
                    for (int i = anchor_coor[0]; i < anchor_coor[2]; ++i)
                    {
                        res = sparse_pillar_map[j * GRID_X_SIZE_ + i];
                        if (res == 1)
                        {
                            _anchor_mask[ind] = 1;
                            break;
                        }
                    }
                    if(res == 1)
                        break;
                }
            }
        }
    }
    return _anchor_mask;
}

void parse_rpn_out(float nms_score_threshold, int cls_size,
                   int boxes_size, int dir_size, int features_size,
                   std::vector<float> cls_data,
                   std::vector<float> box_data,
                   std::vector<float> dir_data,
                   std::vector<int> anchor_mask,
                   std::vector<int> &keep_ids, std::vector<float> &scores,
                   std::vector<int> &cls_argmax, std::vector<int> &dir_cls_argmax,
                   std::vector<std::vector<float>> &boxes)
{
    // 遍历每一个特征
    for (int i = 0; i < features_size; ++i)
    {
        int arg_max_cls = -1;
        float max_cls = 0;
        float nms_score_threshold_cur = nms_score_threshold;
        if(anchor_mask[i] != 1)
            continue;
        for (int k = 0; k < cls_size; ++k)
        {
            int id = i * cls_size + k;
            float cur_cls = cls_data[id];
            // 确定最大得分对应的索引
            if (cur_cls >= nms_score_threshold_cur)
            {
                max_cls = cur_cls;
                arg_max_cls = k;
                nms_score_threshold_cur = cur_cls;
            }
        }
        // 如果当前特征确定为一个信任特征，就输出box与dir
        if (arg_max_cls >= 0)
        {
            keep_ids.push_back(i);
            scores.push_back(max_cls);
            cls_argmax.push_back(arg_max_cls);

            std::vector<float> cur_box(boxes_size);
            for (int k = 0; k < boxes_size; ++k)
            {
                int id = i * boxes_size + k;
                float cur_boxes = box_data[id];
                cur_box[k] = cur_boxes;
            }
            boxes.push_back(cur_box);

            float dir_cls_0 = dir_data[i * dir_size];
            float dir_cls_1 = dir_data[i * dir_size + 1];
            int cur_dir_cls_argmax = (dir_cls_0 > dir_cls_1) ? 0 : 1;
            dir_cls_argmax.push_back(cur_dir_cls_argmax);
        }
    }
}
