#include "pp_base.h"

const float score_threshold_{-0.20067069546215122};
const float nms_overlap_threshold_{0.3};
const int MAX_NUM_PILLARS_{12000};
const int MAX_NUM_POINTS_PER_PILLAR_{100};
const int GRID_X_SIZE_{432};
const int GRID_Y_SIZE_{496};
const int GRID_Z_SIZE_{1};
const int NUM_ANCHOR_X_INDS_{int(GRID_X_SIZE_ * 0.5)};
const int NUM_ANCHOR_Y_INDS_{int(GRID_Y_SIZE_ * 0.5)};
const int NUM_ANCHOR_R_INDS_{2};
const int NUM_ANCHOR_{NUM_ANCHOR_X_INDS_ * NUM_ANCHOR_Y_INDS_ * NUM_ANCHOR_R_INDS_};
const int RPN_BOX_OUTPUT_SIZE_{NUM_ANCHOR_ * 7};
const int RPN_CLS_OUTPUT_SIZE_{NUM_ANCHOR_};
const int RPN_DIR_OUTPUT_SIZE_{NUM_ANCHOR_ * 2};
const float PILLAR_X_SIZE_{0.16};
const float PILLAR_Y_SIZE_{0.16};
const float PILLAR_Z_SIZE_{4.0};
const float MIN_X_RANGE_{0.0};
const float MIN_Y_RANGE_{-39.68};
const float MIN_Z_RANGE_{-3.0};
const float MAX_X_RANGE_{69.12};
const float MAX_Y_RANGE_{39.68};
const float MAX_Z_RANGE_{1.0};
const float SENSOR_HEIGHT_{1.73};
const float ANCHOR_DX_SIZE_{1.6};
const float ANCHOR_DY_SIZE_{3.9};
const float ANCHOR_DZ_SIZE_{1.56};
const int NUM_BOX_CORNERS_{4};
const int NUM_OUTPUT_BOX_FEATURE_{7};
const int NUM_INDS_FOR_SCAN_(512);

class Autoware_PointPillar
{
private:
    /* data */
public:
    Autoware_PointPillar(/* args */);
    ~Autoware_PointPillar();
    void preProcess(const float *in_points_array, const int in_num_points);
    void postProcess();
    
private:
    void pre_process(const float *in_points_array, int in_num_points, int *sparse_pillar_map, int *x_coors, int *y_coors,
                     float *num_points_per_pillar, float *pillar_x, float *pillar_y, float *pillar_z,
                     float *pillar_i, float *x_coors_for_sub_shaped, float *y_coors_for_sub_shaped,
                     float *pillar_feature_mask, int *host_pillar_count);
    void initializeVariables(int *coor_to_pillaridx, float *pillar_x,
                             float *pillar_y, float *pillar_z, float *pillar_i,
                             float *x_coors_for_sub_shaped, float *y_coors_for_sub_shaped);

public:
    int host_pillar_count_[1];

    int *dev_x_coors_;
    int *dev_y_coors_;

    int *dev_sparse_pillar_map_;

    // dayang: input for pfe.onnx
    float *dev_pillar_x_;
    float *dev_pillar_y_;
    float *dev_pillar_z_;
    float *dev_pillar_i_;
    float *dev_num_points_per_pillar_;
    float *dev_x_coors_for_sub_shaped_;
    float *dev_y_coors_for_sub_shaped_;
    float *dev_pillar_feature_mask_;
};

class Autoware_Anchor
{
private:
    const Array3f pc_range_high_{MAX_X_RANGE_, MAX_Y_RANGE_, MAX_Z_RANGE_};
    const Array3f pc_range_low_{MIN_X_RANGE_, MIN_Y_RANGE_, MIN_Z_RANGE_};
    const Array3f voxel_size_{PILLAR_X_SIZE_, PILLAR_Y_SIZE_, PILLAR_Z_SIZE_};
    const int batch_image_height = GRID_Y_SIZE_;
    const int batch_image_width = GRID_X_SIZE_;
    const std::string type_name_ = "Car";
    const Array3f size_{ANCHOR_DX_SIZE_, ANCHOR_DY_SIZE_, ANCHOR_DZ_SIZE_};
    const Array3f stride_{PILLAR_X_SIZE_ * 2, PILLAR_Y_SIZE_ * 2, 0.0};
    const Array3f offset_{MIN_X_RANGE_ + PILLAR_X_SIZE_, MIN_Y_RANGE_ + PILLAR_Y_SIZE_, -SENSOR_HEIGHT_};

public:
    Autoware_Anchor(/* args */);
    ~Autoware_Anchor();

    MatrixXf create_anchor();
    std::vector<int> getAnchorMask(int *sparse_pillar_map);

    MatrixXf _anchors;
    MatrixXf _box_anchors;
    std::vector<int> _anchor_mask;    
};

void parse_rpn_out(float nms_score_threshold, int cls_size,
                   int boxes_size, int dir_size, int features_size,
                   std::vector<float> cls_data,
                   std::vector<float> box_data,
                   std::vector<float> dir_data,
                   std::vector<int> anchor_mask,
                   std::vector<int> &keep_ids, std::vector<float> &scores,
                   std::vector<int> &cls_argmax, std::vector<int> &dir_cls_argmax,
                   std::vector<std::vector<float>> &boxes);
