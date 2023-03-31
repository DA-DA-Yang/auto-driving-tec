
#include "pointpillarsnet.h"
#include <chrono>

PointPillarsNet::PointPillarsNet()
{
    for (int i = 0; i < 3; ++i)
    {
        _offset[i] = pc_range_low[i] + voxel_size[i] / 2.0;
        _voxel_range[i] = floor((pc_range_high[i] - pc_range_low[i]) / voxel_size[i]);
    }
}

vector<float> PointPillarsNet::extract(const PointsInVoxels &piv)
{
    //voxels表示有多少个voxel,每个voxel包含多少个point
    vector<MatrixXf> voxels(std::move(piv.voxels));
    int num_non_empty_voxels = voxels.size();
    MatrixXf points_mean(num_non_empty_voxels, 4);
    for (int i = 0; i < num_non_empty_voxels; ++i)
    {
        //对每个voxels中的point求均值，XYZI
        points_mean.row(i) = voxels[i].colwise().sum() / piv.num_points_per_voxel[i];
        for (int j = 0; j < piv.num_points_per_voxel[i]; ++j)
        {
            //每个point再减去均值
            voxels[i].row(j) -= points_mean.row(i);
        }
    }
    //做一次最大池化，取每个voxel中值最大的point，减去均值后的。
    MatrixXf res_max_pool = max_pool(voxels);
    int num_voxels = _voxel_range[1] * _voxel_range[0];
    vector<float> image1(num_out_features * num_voxels);
    // 执行最大池化，(C, P, N)->(C, P)
    int num_out_features_nn = res_max_pool.cols();
    //遍历非空的voxels
    for (int i = 0; i < num_non_empty_voxels; ++i)
    {
        int cur_coors_idx = piv.coors(i, 1) * _voxel_range[0] + piv.coors(i, 0);
        //输出特征
        for (int j = 0; j < num_out_features; ++j)
        {
            int cur_image1_idx = j * num_voxels + cur_coors_idx;
            //输出XYZI
            if (j < num_out_features_nn)
            {
                image1[cur_image1_idx] = res_max_pool(i, j);
            }
            // 输出Xc,Yc,Zc,Ic
            else if (j < num_out_features_nn + 4)
            {
                image1[cur_image1_idx] = points_mean(i, j - num_out_features_nn);
            }
            else
            {
                //输出pillar的网格坐标*pillar的尺寸+pillar网格原点的实际坐标
                image1[cur_image1_idx] =
                    piv.coors(i, j - (num_out_features_nn + 4)) *
                        voxel_size[j - (num_out_features_nn + 4)] +
                    _offset[j - (num_out_features_nn + 4)];
            }
        }
    }

    return image1;
}

// torch.max(x, dim=1, keepdim=True)[0]
// 在vector<MatrixX4f>的每个MatrixX4f的列维度上取最大
// 返回 (num_non_empty_voxels, num_out_features)
MatrixXf PointPillarsNet::max_pool(const vector<MatrixXf> &input)
{
    int num_non_empty_voxels = input.size();
    MatrixXf res_max_pooled = MatrixXf::Zero(num_non_empty_voxels, 4/*num_out_features - 6*/);
#pragma omp parallel for
    for (int i = 0; i < num_non_empty_voxels; ++i)
    {
        res_max_pooled.block<1, 4 /*num_out_features - 6*/>(i, 0) = input[i].colwise().maxCoeff();
    }
    return res_max_pooled;
}