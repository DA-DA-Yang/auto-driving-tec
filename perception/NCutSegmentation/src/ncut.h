#ifndef NCUT_H
#define NCUT_H

#include "floodfill.h"
#include "define.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Dense"
class NCut
{
private:
    struct GridIndex
    {
        int irow;
        int jcol;
    };
    // x_min, x_max, y_min, y_max, z_min, z_max;
  typedef std::tuple<float, float, float, float, float, float> NCutBbox;

public:
    NCut(/* args */);
    ~NCut();

    /**
     *@brief 根据输入点云，获得点云中的连通域
     *@param pointcloud 输入点云
     */
    void segments(const std::vector<PointXYZI> *pointcloud);

    // 返回连通域的数目
    inline int getNumSegments() const { return static_cast<int>(_segment_pids.size()); }
    // 返回连通域点的索引
    inline const std::vector<std::vector<int>> &getPointIndicesInSegments() { return _segment_pids; }
    // 返回连通域的标签
    inline const std::vector<std::string> &getLabels() const { return _segment_labels; }
    // 返回连通域的bbox
    inline const std::vector<NCutBbox> &getBboxes() const { return _segment_bbox; }

private:
    void _superPixelFloodFill(const std::vector<PointXYZI> *pointcloud, float grid_radius, float cell_size, std::vector<std::vector<int>> *super_pixels);
    void _preComputeSkeletonAndBbox();
    void _buildAverageHeightMap(const std::vector<PointXYZI> *pointcloud, FloodFill ff, cv::Mat *cv_height_map, std::vector<GridIndex> *point_pixels);
    void _sampleByGrid(const std::vector<int>& point_gids, Eigen::MatrixXf* skeleton_coords, Eigen::MatrixXf* skeleton_features);
    void _normalizedCut(float ncut_threshold, bool use_classifier, std::vector<std::vector<int>> &segment_clusters, std::vector<std::string> &segment_labels);
    void _computeSkeletonWeights(Eigen::MatrixXf *weights);
    bool _computeSquaredSkeletonDistance(const Eigen::MatrixXf &in1_points, const Eigen::MatrixXf &in1_features,
                                         const Eigen::MatrixXf &in2_points, const Eigen::MatrixXf &in2_features,
                                         float *dist_point, float *dist_feature);
    float _getMinNcuts(const Eigen::MatrixXf &in_weights,
                        const std::vector<int> *in_clusters,
                        std::vector<int> *seg1, std::vector<int> *seg2);      
    void _laplacianDecomposition(const Eigen::MatrixXf &weights, Eigen::MatrixXf *eigenvectors_in);
    void _getClustersPointIds(const std::vector<int> &cluster_ids, std::vector<int> *point_ids);
    NCutBbox _computeBbox(const std::vector<int> &point_gids);
    int _getBbox(const std::vector<int> &cluster_ids, NCutBbox *box);

private:
    std::vector<PointXYZI> *_ori_pointcloud;

    float _grid_radius{80.f};
    float _super_pixel_cell_size{0.25f};
    // 经过superpixel分割后的连通域
    std::vector<std::vector<int>> _point_index_in_super_pixels;

    FloodFill _ff_skeleton;
    float _skeleton_cell_size{0.2f};
    // 网格的平均高度图
    cv::Mat _cv_height_map;
    // 提取高度特征的尺寸
    int _patch_size{3};
    // 每个连通域内包含网格的平均坐标
    std::vector<Eigen::MatrixXf> _skeleton_points_in_super_pixels;
    // 每个连通域内包含网格的高度特征
    std::vector<Eigen::MatrixXf> _skeleton_features_in_super_pixels;
    // 每个连通域的bbox
    std::vector<NCutBbox> _bboxes;
    // 每个连通域的标签
    std::vector<std::string> _labels;

    float _ncut_threshold{0.4f};
    float _sigma_feature{1.5f};
    float _sigma_space{1.5f};
    float _connect_radius{1.0f};
    int _num_cuts{5};

    // final segments, each vector contains
    std::vector<std::vector<int>> _segment_pids;
    std::vector<std::string> _segment_labels;
    std::vector<NCutBbox> _segment_bbox;
    std::vector<std::vector<int>> _outlier_pids;
};

#endif