#include <memory>
#include "pp_base.h"

typedef struct PointXYZI ///< user defined point type
{
    float x;
    float y;
    float z;
    float intensity;
} PointXYZI;

class pfe_process
{
public:
    //std::shared_ptr<signed char[]> int_buf_ = nullptr;
    std::shared_ptr<float[]> float_buf_ = nullptr;
    int input_w_ = batch_image_width;
    int input_h_ = batch_image_height;
    int input_c_ = num_out_features;

public:
    pfe_process(/* args */);
    ~pfe_process();
    void process(void *ptr, int size);
};
