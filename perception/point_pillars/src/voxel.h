
#ifndef VOXEL_H
#define VOXEL_H
#include <cstddef>
#include <vector>
#include "pp_base.h"

struct PointsInVoxels
{
    vector<MatrixXf> voxels;
    MatrixXi coors;
    vector<int> num_points_per_voxel;

    PointsInVoxels(vector<MatrixXf> &&voxelsi,
                   MatrixXi &&coorsi,
                   vector<int> &&num_points_per_voxeli)
        : voxels(std::move(voxelsi)), coors(std::move(coorsi)), num_points_per_voxel(std::move(num_points_per_voxeli)) {}

    PointsInVoxels(PointsInVoxels &&other) noexcept;
    PointsInVoxels &operator=(PointsInVoxels &&other) noexcept;
};

class Voxel
{
private:
    Array3i _voxel_range;

public:
    Voxel();
    PointsInVoxels points_to_voxels(const MatrixXf &point_cloud);
};

#endif