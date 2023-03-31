
#include <string>
#include "pp_base.h"

class AnchorGenerator
{
public:
    AnchorGenerator(const std::string &class_name,
                    const Eigen::Array3f &stride,
                    const Eigen::Array3f &size,
                    const Eigen::Array3f &offset);

    Eigen::MatrixXf get_anchors();

private:
    std::string _class_name;
    Eigen::MatrixXf _anchors;
};

Eigen::MatrixXf create_anchor();
