
#ifndef POINTPILLARSNET_H
#define POINTPILLARSNET_H

#include <omp.h>
#include <fstream>
#include "voxel.h"

class PointPillarsNet
{
private:
    MatrixXf max_pool(const vector<MatrixXf> &input);

    Array3f _offset;
    Array3i _voxel_range;

public:
    PointPillarsNet();
    vector<float> extract(const PointsInVoxels &piv);
};

#endif