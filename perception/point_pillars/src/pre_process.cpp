
#include "pre_process.h"

#include <chrono>

MatrixXf read_matrix(const string &path, int rows, int cols, bool is_trans)
{

    if (cols < 0)
    {
        throw invalid_argument("cols >= 0");
    }

    ifstream file(path, ios::in | ios::binary);
    if (!file)
    {
        throw invalid_argument("no such file");
    }

    int begin = file.tellg();
    file.seekg(0, std::ios::end);
    int end = file.tellg();
    size_t len = end - begin;
    size_t data_len = len / sizeof(float);

    if (-1 == rows)
    {
        rows = data_len / cols;
    }

    MatrixXf res(cols, rows);
    ifstream in(path, ios::in | ios::binary);
    in.read((char *)res.data(), rows * cols * sizeof(float));
    in.close();
    if (is_trans)
    {
        return res.transpose();
    }
    else
    {
        return res;
    }
    
}

vector<float> pre_process(const string &points_path)
{
    MatrixXf points(std::move(read_matrix(points_path, -1, 4, true)));
    Voxel voxel = Voxel();
    PointsInVoxels piv(std::move(voxel.points_to_voxels(points)));


    PointPillarsNet ppn;

    vector<float> batch_image = ppn.extract(piv);

 
    return batch_image;
}
