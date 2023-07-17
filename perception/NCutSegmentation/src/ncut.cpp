
#include "ncut.h"

NCut::NCut(/* args */)
{
    _ori_pointcloud = new std::vector<PointXYZI>;
}

NCut::~NCut()
{
}

void NCut::segments(const std::vector<PointXYZI> *pointcloud)
{
    // 每个点在点云中的索引
    std::vector<int> pids(pointcloud->size());
    for (size_t i = 0; i < pointcloud->size(); ++i)
    {
        pids[i] = static_cast<int>(i);
    }

    // 保存原始点云
    int num_points = pointcloud->size();
    _ori_pointcloud->resize(pointcloud->size());
    for (int i = 0; i < num_points; ++i)
    {
        _ori_pointcloud->push_back((*pointcloud)[i]);
    }

    // 进行super pixels的网格建立
    _superPixelFloodFill(pointcloud, _grid_radius, _super_pixel_cell_size, &_point_index_in_super_pixels);

    // 计算skeleton特征与边界框
    _preComputeSkeletonAndBbox();

    std::vector<std::vector<int>> segment_clusters;
    std::vector<std::string> segment_labels;
    _normalizedCut(_ncut_threshold, true, segment_clusters, segment_labels);

    _segment_pids.clear();
    _segment_labels.clear();
    for (size_t i = 0; i < segment_clusters.size(); ++i)
    {
        std::vector<int> pids;
        _getClustersPointIds(segment_clusters[i], &pids);
        if (pids.size() > 0)
        {
            _segment_pids.push_back(pids);
            _segment_labels.push_back(segment_labels[i]);
            NCutBbox box;
            _getBbox(segment_clusters[i], &box);
            _segment_bbox.push_back(box);
        }
    }
}

void NCut::_superPixelFloodFill(const std::vector<PointXYZI> *pointcloud, float grid_radius, float cell_size, std::vector<std::vector<int>> *super_pixels)
{
    FloodFill spff(grid_radius, cell_size);
    std::vector<int> num_cells_per_component;
    spff.getSegments(*pointcloud, *super_pixels, num_cells_per_component);
}

void NCut::_preComputeSkeletonAndBbox()
{
    const int num_clusters = static_cast<int>(_point_index_in_super_pixels.size());

    // 进行skeleton的网格建立
    _ff_skeleton.setGridRadius(_grid_radius);
    _ff_skeleton.setCellSize(_skeleton_cell_size);
    _ff_skeleton.buildGrid(*_ori_pointcloud);
    // 计算skeleton网格平均高度图
    std::vector<GridIndex> point_pixels;
    _buildAverageHeightMap(_ori_pointcloud, _ff_skeleton, &_cv_height_map, &point_pixels);

    //对每个经过super pixel分割的连通域进行采样，并计算特征
    _skeleton_points_in_super_pixels.resize(num_clusters);
    _skeleton_features_in_super_pixels.resize(num_clusters);
    _bboxes.resize(num_clusters);
    _labels.resize(num_clusters);
    for (int i = 0; i < num_clusters; ++i)
    {
        // 把经过super pixel分割的连通域，再通过skeleton建立网格，进行采样与高度特征计算
        _sampleByGrid(_point_index_in_super_pixels[i], &_skeleton_points_in_super_pixels[i], &_skeleton_features_in_super_pixels[i]);
        _bboxes[i] = _computeBbox(_point_index_in_super_pixels[i]);
        _labels.push_back("unkown");
    }
}

void NCut::_buildAverageHeightMap(const std::vector<PointXYZI> *pointcloud, FloodFill ff, cv::Mat *cv_height_map, std::vector<GridIndex> *point_pixels)
{
    const int num_points = static_cast<int>(pointcloud->size());
    // 创建高度图，大小与floodfill的网格一致
    cv::Mat height_map_tmp = cv::Mat::zeros(ff.getNumRows(), ff.getNumCols(), CV_32F);
    // 高度图每个格子包含的点云数量
    cv::Mat num_points_map = cv::Mat::zeros(ff.getNumRows(), ff.getNumCols(), CV_16U);

    (*point_pixels).resize(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        int irow = -1;
        int jcol = -1;
        PointXYZI pt = (*pointcloud)[i];
        if (ff.getGridIndex(pt.x, pt.y, irow, jcol))
        {
            height_map_tmp.at<float>(irow, jcol) += pt.z;
            num_points_map.at<int>(irow, jcol) += 1;
            (*point_pixels)[i].irow = irow;
            (*point_pixels)[i].jcol = jcol;
        }
        else
        {
            (*point_pixels)[i].irow = -1;
            (*point_pixels)[i].jcol = -1;
        }
    }
    for (int i = 0; i < height_map_tmp.rows; ++i)
    {
        for (int j = 0; j < height_map_tmp.cols; ++j)
        {
            if (num_points_map.at<int>(i, j) > 0)
            {
                height_map_tmp.at<float>(i, j) /= static_cast<float>(num_points_map.at<int>(i, j));
            }
        }
    }
    cv::Mat height_map_tmp_copy;
    // 高斯滤波
    cv::GaussianBlur(height_map_tmp, height_map_tmp_copy, cv::Size(3, 3), 0, 0);
    // 归一化到0-255区间
    cv::normalize(height_map_tmp_copy, *cv_height_map, 0, 255, cv::NORM_MINMAX,
                  CV_8UC1);
}

void NCut::_sampleByGrid(const std::vector<int> &point_gids, Eigen::MatrixXf *skeleton_coords, Eigen::MatrixXf *skeleton_features)
{
    // 根据点的索引，从原始点云中拷贝
    int num_points = static_cast<int>(point_gids.size());
    std::vector<PointXYZI> pc;
    pc.resize(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        pc.push_back((*_ori_pointcloud)[point_gids[i]]);
    }

    // 建立网格
    FloodFill sampler(_grid_radius, _skeleton_cell_size);
    sampler.buildGrid(pc);
    const std::vector<int> &grid_index_in_pc = sampler.getPointInGridIdx();
    // 点云所占网格数目
    int num_grids = static_cast<int>(grid_index_in_pc.size());

    // 建立查找表，<网格索引，<点的累计坐标，点的个数>>
    std::unordered_map<int, std::pair<PointXYZI, float>> grid_points;
    std::unordered_map<int, std::pair<PointXYZI, float>>::iterator ite;
    // 对每个网格进行统计
    for (int i = 0; i < num_grids; ++i)
    {
        // 查找网格内的点
        ite = grid_points.find(grid_index_in_pc[i]);
        if(ite != grid_points.end())
        {
            // 如果该网格内存在点，就把点的坐标累计，同时计数加1
            ite->second.first.x += pc[i].x;
            ite->second.first.y += pc[i].y;
            ite->second.first.z += pc[i].z;
            ite->second.first.i += pc[i].i;
            ite->second.second += 1.f;
        }
        else
        {
            // 如果该网格内不存在点，就把当前点存进去
            grid_points[grid_index_in_pc[i]] = std::make_pair(pc[i], 1.f);
        }
    }

    // 对每个网格坐标求平均均值
    int num_coords = static_cast<int>(grid_points.size());
    skeleton_coords->resize(num_coords, 3);
    int count = 0;
    for (ite = grid_points.begin(); ite != grid_points.end(); ++ite)
    {
        skeleton_coords->coeffRef(count, 0) = ite->second.first.x / ite->second.second;
        skeleton_coords->coeffRef(count, 1) = ite->second.first.y / ite->second.second;
        skeleton_coords->coeffRef(count, 2) = ite->second.first.z / ite->second.second;
        count++;
    }

    // 计算skeleton feature,就是每个点邻域的9个高度值
    skeleton_features->resize(num_coords, _patch_size * _patch_size);
    for (int i = 0; i < num_coords; ++i)
    {
        int row = 0, col = 0;
        // 获取网格的坐标
        _ff_skeleton.getGridIndex(skeleton_coords->coeffRef(i, 0), skeleton_coords->coeffRef(i, 0), row, col);
        cv::Mat patch;
        cv::Point2f pt((float)row, (float)col);
        // 从高度图中提取3*3的高度特征
        cv::getRectSubPix(_cv_height_map, cv::Size(_patch_size, _patch_size), pt, patch);
        int npatch = 0;
        for (int r = 0; r < patch.rows; ++r)
            for (int c = 0; c < patch.cols;++c)
            {
                float val = patch.at<float>(r, c);
                // 限制其不为无效值
                val = static_cast<float>((std::isnan(val) || std::isinf(val)) ? 1.e-50 : val);
                skeleton_features->coeffRef(i, npatch++) = val;
            }
    }
}

NCut::NCutBbox NCut::_computeBbox(const std::vector<int> &point_gids)
{
    // 根据给的点云索引，计算该点云的bounding box
    // ! Note: do not perform rotation, so just some intuitive guess
    float x_max = -std::numeric_limits<float>::max();
    float y_max = -std::numeric_limits<float>::max();
    float z_max = -std::numeric_limits<float>::max();
    float x_min = std::numeric_limits<float>::max();
    float y_min = std::numeric_limits<float>::max();
    float z_min = std::numeric_limits<float>::max();
    for (size_t j = 0; j < point_gids.size(); ++j)
    {
        int pid = point_gids[j];
        x_min = std::min(x_min, (*_ori_pointcloud)[pid].x);
        x_max = std::max(x_max, (*_ori_pointcloud)[pid].x);
        y_min = std::min(y_min, (*_ori_pointcloud)[pid].y);
        y_max = std::max(y_max, (*_ori_pointcloud)[pid].y);
        z_min = std::min(z_min, (*_ori_pointcloud)[pid].z);
        z_max = std::max(z_max, (*_ori_pointcloud)[pid].z);
    }
    NCut::NCutBbox box;
    std::get<0>(box) = x_min;
    std::get<1>(box) = x_max;
    std::get<2>(box) = y_min;
    std::get<3>(box) = y_max;
    std::get<4>(box) = z_min;
    std::get<5>(box) = z_max;
    return box;
}

int NCut::_getBbox(const std::vector<int> &cluster_ids, NCut::NCutBbox *box)
{
    if (cluster_ids.empty())
    {
        return 0;
    }
    int cid = cluster_ids[0];
    float x_min = std::get<0>(_bboxes[cid]);
    float x_max = std::get<1>(_bboxes[cid]);
    float y_min = std::get<2>(_bboxes[cid]);
    float y_max = std::get<3>(_bboxes[cid]);
    float z_min = std::get<4>(_bboxes[cid]);
    float z_max = std::get<5>(_bboxes[cid]);
    int num_points = static_cast<int>(_point_index_in_super_pixels[cid].size());
    for (size_t i = 1; i < cluster_ids.size(); ++i)
    {
        cid = cluster_ids[i];
        x_min = std::min(x_min, std::get<0>(_bboxes[cid]));
        x_max = std::max(x_max, std::get<1>(_bboxes[cid]));
        y_min = std::min(y_min, std::get<2>(_bboxes[cid]));
        y_max = std::max(y_max, std::get<3>(_bboxes[cid]));
        z_min = std::min(y_min, std::get<4>(_bboxes[cid]));
        z_max = std::max(y_max, std::get<5>(_bboxes[cid]));
        num_points += static_cast<int>(_point_index_in_super_pixels[cid].size());
    }
    std::get<0>(*box) = x_min;
    std::get<1>(*box) = x_max;
    std::get<2>(*box) = y_min;
    std::get<3>(*box) = y_max;
    std::get<4>(*box) = z_min;
    std::get<5>(*box) = z_max;
    return num_points;
}

void NCut::_normalizedCut(float ncut_threshold, bool use_classifier, std::vector<std::vector<int>> &segment_clusters, std::vector<std::string> &segment_labels)
{
    const int num_clusters = static_cast<int>(_point_index_in_super_pixels.size());
    if (num_clusters < 1)
    {
        return;
    }
    if (num_clusters == 1)
    {
        std::vector<int> tmp(1, 0);
        segment_clusters.push_back(tmp);
        segment_labels.push_back(_labels[0]);
        return;
    }

    // 计算skeleton权重
    Eigen::MatrixXf weights;
    _computeSkeletonWeights(&weights);

    // 经过super pixel分割的连通域的序号
    std::vector<int> *curr = new std::vector<int>(num_clusters);
    for (int i = 0; i < num_clusters; ++i)
    {
        (*curr)[i] = i;
    }
    // 放入堆栈中
    std::stack<std::vector<int> *> job_stack;
    job_stack.push(curr);

    while (!job_stack.empty())
    {
        // 提取出堆栈中待处理的连通域
        curr = job_stack.top();
        job_stack.pop();

        std::string seg_label;
        if (curr->size() == 1)
        {
            segment_clusters.push_back(*curr);
            segment_labels.push_back(_labels[(*curr)[0]]);
        }
        else if (use_classifier && false/*IsMovableObstacle(*curr, &seg_label)*/)
        {
            segment_clusters.push_back(*curr);
            segment_labels.push_back(seg_label);
        }
        else
        {
            std::vector<int> *seg1 = new std::vector<int>();
            std::vector<int> *seg2 = new std::vector<int>();

            // 提取连通域两两之间的权重系数
            Eigen::MatrixXf my_weights(curr->size(), curr->size());
            for (size_t i = 0; i < curr->size(); ++i)
            {
                const int ci = curr->at(i);
                for (size_t j = 0; j < curr->size(); ++j)
                {
                    const int cj = curr->at(j);
                    my_weights.coeffRef(i, j) = weights.coeffRef(ci, cj);
                }
            }

            // 把一个集合curr分成两个集合seg1与seg2，同时输出两个集合之间的ncut cost
            double cost = _getMinNcuts(my_weights, curr, seg1, seg2);

            if (cost > _ncut_threshold || 0 == seg1->size() || 0 == seg2->size())
            {
                // 已经分割到最小了
                std::vector<int> buffer;
                for (size_t i = 0; i < curr->size(); ++i)
                {
                    const int cid = (*curr)[i];
                    if (_labels[cid] != "unknown")
                    {
                        std::vector<int> tmp(1, cid);
                        segment_clusters.push_back(tmp);
                        segment_labels.push_back(_labels[cid]);
                    }
                    else
                    {
                        buffer.push_back(cid);
                    }
                }
                if (buffer.size() > 0)
                {
                    // 保存最小分割后的连通域及其标签
                    segment_clusters.push_back(buffer);
                    segment_labels.push_back("unknown");
                }
                delete seg1;
                delete seg2;
            }
            else
            {
                // 还可以继续分割的，就放入堆栈中
                job_stack.push(seg1);
                job_stack.push(seg2);
            }
        }
    } // end of while
}

void NCut::_computeSkeletonWeights(Eigen::MatrixXf *weights)
{
    const int num_clusters = static_cast<int>(_point_index_in_super_pixels.size());

    const double hs2 = _sigma_space * _sigma_space;
    const double hf2 = _sigma_feature * _sigma_feature;
    const double radius2 = _connect_radius * _connect_radius;

    weights->resize(num_clusters, num_clusters);

    for (int i = 0; i < num_clusters; ++i)
    {
        weights->coeffRef(i, i) = 1.f;
        for (int j = i + 1; j < num_clusters; ++j)
        {
            float dist_point = std::numeric_limits<float>::max();
            float dist_feature = std::numeric_limits<float>::max();
            _computeSquaredSkeletonDistance(
                _skeleton_points_in_super_pixels[i], _skeleton_features_in_super_pixels[i],
                _skeleton_points_in_super_pixels[j], _skeleton_features_in_super_pixels[j],
                &dist_point, &dist_feature);
            if(dist_point > radius2)
            {
                // 如果两个连通域的欧式距离大于connect_radius,表明两个连通域没有关联性
                weights->coeffRef(i, j) = 0.f;
                weights->coeffRef(j, i) = 0.f;
            }
            else
            {
                weights->coeffRef(i, j) = static_cast<float>(exp(-dist_point / hs2) *
                                                            exp(-dist_feature / hf2));
                weights->coeffRef(j, i) = weights->coeffRef(i, j);
            }
        }
    }
}

bool NCut::_computeSquaredSkeletonDistance(const Eigen::MatrixXf &in1_points, const Eigen::MatrixXf &in1_features,
                                     const Eigen::MatrixXf &in2_points, const Eigen::MatrixXf &in2_features,
                                     float *dist_point, float *dist_feature)
{
    // 计算两个点集之间的欧式距离的平方，计算两个feature之间的特征距离的平方
    if (!((in1_points.rows() == in1_features.rows()) &&
          (in2_points.rows() == in2_features.rows())))
    {
        return false;
    }
    const int num1 = static_cast<int>(in1_points.rows());
    const int num2 = static_cast<int>(in2_points.rows());
    const int dim = static_cast<int>(in1_features.cols());
    int min_index1 = -1;
    int min_index2 = -1;
    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i < num1; ++i)
    {
        for (int j = 0; j < num2; ++j)
        {
            const float diff_x =
                in1_points.coeffRef(i, 0) - in2_points.coeffRef(j, 0);
            const float diff_y =
                in1_points.coeffRef(i, 1) - in2_points.coeffRef(j, 1);
            const float diff_z =
                in1_points.coeffRef(i, 2) - in2_points.coeffRef(j, 2);
            float dist = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
            if (dist < min_dist)
            {
                min_dist = dist;
                min_index1 = i;
                min_index2 = j;
            }
        }
    }
    *dist_point = min_dist;
    *dist_feature = 0.f;
    for (int i = 0; i < dim; ++i)
    {
        float diff = in1_features.coeffRef(min_index1, i) -
                     in2_features.coeffRef(min_index2, i);
        *dist_feature += diff * diff;
    }
    return true;
}

float NCut::_getMinNcuts(const Eigen::MatrixXf &in_weights,
                   const std::vector<int> *in_clusters,
                   std::vector<int> *seg1, std::vector<int> *seg2)
{
     // .0 initialization
    const int num_clusters = static_cast<int>(in_weights.rows());
    seg1->resize(num_clusters);
    seg2->resize(num_clusters);

    // .1 eigen decompostion
    Eigen::MatrixXf eigenvectors;
    _laplacianDecomposition(in_weights, &eigenvectors);

    // .2 search for best split
    const float minval = eigenvectors.col(1).minCoeff();
    const float maxval = eigenvectors.col(1).maxCoeff();
    const float increment = static_cast<float>(
        (maxval - minval) / (static_cast<float>(_num_cuts) + 1.0f));
    int num_seg1 = 0;
    int num_seg2 = 0;
    float opt_split = 0.0;
    float opt_cost = std::numeric_limits<float>::max();
    for (int i = 0; i < _num_cuts; ++i) {
        num_seg1 = 0;
        num_seg2 = 0;
        // .2.1 split
        float split =
            static_cast<float>(minval + static_cast<float>(i + 1) * increment);
        for (int j = 0; j < num_clusters; ++j) {
        if (eigenvectors.coeffRef(j, 1) > split) {
            (*seg1)[num_seg1++] = j;
        } else {
            (*seg2)[num_seg2++] = j;
        }
        }
        // .2.2 compute best normalized_cuts cost
        double assoc1 = 0.0;
        double assoc2 = 0.0;
        double cut = 0.0;
        for (int j = 0; j < num_seg1; ++j) {
            assoc1 += in_weights.row(seg1->at(j)).sum();
        }
        for (int j = 0; j < num_seg2; ++j) {
            assoc2 += in_weights.row(seg2->at(j)).sum();
        }
        for (int j = 0; j < num_seg1; ++j) {
            for (int t = 0; t < num_seg2; ++t) {
                cut += in_weights.coeffRef(seg1->at(j), seg2->at(t));
        }
        }
        float cost = static_cast<float>(cut / assoc1 + cut / assoc2);

        // .2.3 find best cost
        if (cost < opt_cost) {
            opt_cost = cost;
            opt_split = split;
        }
    }
    // .3 split data according to best split
    num_seg1 = 0;
    num_seg2 = 0;
    for (int i = 0; i < num_clusters; ++i) {
        if (eigenvectors.coeffRef(i, 1) > opt_split) {
            (*seg1)[num_seg1++] = in_clusters->at(i);
         } else {
            (*seg2)[num_seg2++] = in_clusters->at(i);
        }
    }
    seg1->resize(num_seg1);
    seg2->resize(num_seg2);
    return opt_cost;
}

void NCut::_laplacianDecomposition(const Eigen::MatrixXf &weights, Eigen::MatrixXf *eigenvectors_in)
{
    Eigen::MatrixXf &eigenvectors = *eigenvectors_in;
  // .1 degree matrix: D = sum(W, 2)
  Eigen::VectorXf diag(weights.rows());
  for (int i = 0; i < weights.rows(); ++i) {
    diag.coeffRef(i) = weights.row(i).sum();
  }
  // .2 graph laplacian L = D - W
  Eigen::MatrixXf laplacian(weights.rows(), weights.cols());
  for (int i = 0; i < laplacian.rows(); ++i) {
    for (int j = 0; j < laplacian.cols(); ++j) {
      if (i == j) {
        laplacian.coeffRef(i, j) = diag.coeffRef(i) - weights.coeffRef(i, j);
      } else {
        laplacian.coeffRef(i, j) = 0.f - weights.coeffRef(i, j);
      }
    }
  }

  // .3 D^(-1/2)
  Eigen::VectorXf diag_halfinv(weights.rows());
  for (int i = 0; i < weights.rows(); ++i) {
    diag_halfinv.coeffRef(i) =
        static_cast<float>(1.0 / std::sqrt(diag.coeffRef(i)));
  }
  // .4 normalized laplacian D^(-1/2) * L * D^(-1/2)
  for (int i = 0; i < laplacian.rows(); ++i) {
    laplacian.row(i) *= diag_halfinv.coeffRef(i);
  }
  for (int j = 0; j < laplacian.cols(); ++j) {
    laplacian.col(j) *= diag_halfinv.coeffRef(j);
  }
  // .4.2 for numerical stability, add eps to the diagonal of laplacian
  float eps = 1e-10f;
  for (int i = 0; i < laplacian.rows(); ++i) {
    laplacian.coeffRef(i, i) += eps;
  }

  // .5 solve eigen decompostion: TODO: lanczos
  Eigen::EigenSolver<Eigen::MatrixXf> eig_solver(laplacian);

  // .6 sort eigen values
  std::vector<std::pair<float, int>> eigval(laplacian.rows());
  for (size_t i = 0; i < eigval.size(); ++i) {
    eigval[i] = std::make_pair(eig_solver.eigenvalues()[i].real(), i);
  }
  std::sort(eigval.begin(), eigval.end(), std::less<std::pair<float, int>>());
  // .7 get sorted eigen vectors
  eigenvectors.resize(weights.rows(), weights.cols());
  for (int i = 0; i < eigenvectors.cols(); ++i) {
    eigenvectors.col(i) =
        eig_solver.eigenvectors().col(eigval[i].second).real();
  }
  for (int i = 0; i < eigenvectors.rows(); ++i) {
    eigenvectors.row(i) *= diag_halfinv.coeffRef(i);
  }
}

void NCut::_getClustersPointIds(const std::vector<int> &cluster_ids, std::vector<int> *point_ids)
{
  std::vector<int> &pids = *point_ids;
  int num_points = 0;
  for (size_t i = 0; i < cluster_ids.size(); ++i)
  {
    num_points += static_cast<int>(_point_index_in_super_pixels[cluster_ids[i]].size());
  }
  pids.resize(num_points, -1);
  int offset = 0;
  for (size_t i = 0; i < cluster_ids.size(); ++i)
  {
    const std::vector<int> &curr_pids = _point_index_in_super_pixels[cluster_ids[i]];
    memcpy(pids.data() + offset, curr_pids.data(),
           sizeof(int) * curr_pids.size());
    offset += static_cast<int>(curr_pids.size());
  }
}