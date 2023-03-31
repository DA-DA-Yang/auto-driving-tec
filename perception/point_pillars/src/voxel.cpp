
#include <math.h>
#include "voxel.h"
#include <chrono>
#include <iostream>
#include <fstream>

Voxel::Voxel()
{
    for (int i = 0; i < 3; ++i)
    {
        _voxel_range[i] = floor((pc_range_high[i] - pc_range_low[i]) / voxel_size[i]);
    }
}

PointsInVoxels::PointsInVoxels(PointsInVoxels &&other) noexcept
{
    // cout << "PointsInVoxels move copy" << endl;
    voxels = std::move(other.voxels);
    coors = std::move(other.coors);
    num_points_per_voxel = std::move(other.num_points_per_voxel);
}

PointsInVoxels &PointsInVoxels::operator=(PointsInVoxels &&other) noexcept
{
    // cout << "move assignment" << endl;
    if (this != &other)
    {
        voxels = std::move(other.voxels);
        coors = std::move(other.coors);
        num_points_per_voxel = std::move(other.num_points_per_voxel);
    }
    return *this;
}

PointsInVoxels Voxel::points_to_voxels(const MatrixXf &point_cloud)
{
    // tmp_coors记录坐标到voxel_idx的映射。初始化为-1
    vector<int> tt(_voxel_range[0] * _voxel_range[1] * _voxel_range[2], -1);
    vector<int> tmp_coors(_voxel_range[0] * _voxel_range[1] * _voxel_range[2], -1);

    int voxel_num = 0;
    vector<MatrixXf> voxels(max_voxels);

    MatrixXi coors = MatrixXi::Zero(max_voxels, 3);
    vector<int> num_points_per_voxel(max_voxels); // 初始化_max_voxels个，初始值为0

    for (int r = 0; r < point_cloud.rows(); ++r)
    {
        auto p = point_cloud.row(r);
        Array3i c;
        bool in_range = true;
        // 计算当前点落在voxel的坐标
        // 如果落在pc_range之外，则直接舍弃
        for (int i = 0; i < 3; ++i)
        {
            c[i] = floor((p[i] - pc_range_low[i]) / voxel_size[i]);

            if ((c[i] < 0) || (c[i] >= _voxel_range[i]))
            {
                in_range = false;
                break;
            }
        }
        if (!in_range)
        {
            continue;
        }

        // int voxel_idx = tmp_coors[c[0]][c[1]][c[2]];
        int voxel_idx = tmp_coors[c[2] + c[1] * _voxel_range[2] + c[0] * _voxel_range[1] * _voxel_range[2]];
        if (-1 == voxel_idx) // 表示当前第一次有点落在voxel_idx索引的这个voxel
        {
            voxel_idx = voxel_num;
            voxels[voxel_idx] = MatrixXf::Zero(max_points_per_voxel, 4);
            if (voxel_num >= max_voxels)
            {
                break;
            }
            voxel_num += 1;
            // tmp_coors[c[0]][c[1]][c[2]] = voxel_idx;
            tmp_coors[c[2] + c[1] * _voxel_range[2] + c[0] * _voxel_range[1] * _voxel_range[2]] = voxel_idx;
            coors.row(voxel_idx) = c;
        }
        int num = num_points_per_voxel[voxel_idx];
        if (num < max_points_per_voxel)
        {
            num_points_per_voxel[voxel_idx] += 1;
            voxels[voxel_idx].row(num) = p;
        }
    }
    for (int i = voxel_num; i < max_voxels; ++i) // 去掉尾部
    {
        num_points_per_voxel.pop_back();
        voxels.pop_back();
    }
    coors.conservativeResize(voxel_num, 3); // 去掉尾部
    PointsInVoxels piv(std::move(voxels), std::move(coors), std::move(num_points_per_voxel));

    return piv;
}
