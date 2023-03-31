#ifndef PRE_PROCESS_H
#define PRE_PROCESS_H

#include "pp_base.h"
#include "voxel.h"
#include "pointpillarsnet.h"

MatrixXf read_matrix(const string &path, int rows, int cols, bool is_trans = false);

vector<float> pre_process(const string &points_path);

#endif