
#include "anchor.h"
#include <math.h>

AnchorGenerator::AnchorGenerator(const std::string &class_name,
                                 const Array3f &stride,
                                 const Array3f &size,
                                 const Array3f &offset) : _class_name(class_name)
{
    Array3i xyz_range;
    for (int i = 0; i < 2; ++i)
    {
        xyz_range[i] = floor((pc_range_high[i] - pc_range_low[i]) / stride[i]);
    }
    xyz_range[2] = 1; 

    _anchors = Eigen::MatrixXf(xyz_range[0] * xyz_range[1] * xyz_range[2] * 2, 7);

    for (int z = 0; z < xyz_range[2]; ++z)
    {
        for (int y = 0; y < xyz_range[1]; ++y)
        {
            for (int x = 0; x < xyz_range[0]; ++x)
            {
                int cur_id = 2 * (x + y * xyz_range[0] + z * xyz_range[0] * xyz_range[1]);

                _anchors.row(cur_id) << offset[0] + x * stride[0],
                    offset[1] + y * stride[1],
                    offset[2] + z * stride[2],
                    size[0],
                    size[1],
                    size[2],
                    0;

                _anchors.row(cur_id + 1) << offset[0] + x * stride[0],
                    offset[1] + y * stride[1],
                    offset[2] + z * stride[2],
                    size[0],
                    size[1],
                    size[2],
                    // M_PI / 2;
                    1.57;
            }
        }
    }
}

Eigen::MatrixXf AnchorGenerator::get_anchors()
{
    return _anchors;
}

Eigen::MatrixXf create_anchor()
{
    AnchorGenerator Car(class_name_Car, stride_Car, size_Car, offset_Car);
    Eigen::MatrixXf anchor_Car = std::move(Car.get_anchors());

    AnchorGenerator Pedestrian(class_name_Pedestrian, stride_Pedestrian, size_Pedestrian, offset_Pedestrian);
    Eigen::MatrixXf anchor_Pedestrian = std::move(Pedestrian.get_anchors());

    AnchorGenerator Cyclist(class_name_Cyclist, stride_Cyclist, size_Cyclist, offset_Cyclist);
    Eigen::MatrixXf anchor_Cyclist = std::move(Cyclist.get_anchors());

    AnchorGenerator Van(class_name_Van, stride_Van, size_Van, offset_Van);
    Eigen::MatrixXf anchor_Van = std::move(Van.get_anchors());

    int num_anchors = anchor_Car.rows() + anchor_Pedestrian.rows() + anchor_Cyclist.rows() + anchor_Van.rows();
    Eigen::MatrixXf anchors(num_anchors, box_coder_size);
    int start_row = 0;
    anchors.block(start_row, 0, anchor_Car.rows(), box_coder_size) = std::move(anchor_Car);
    start_row += anchor_Car.rows();
    anchors.block(start_row, 0, anchor_Pedestrian.rows(), box_coder_size) = std::move(anchor_Pedestrian);
    start_row += anchor_Pedestrian.rows();
    anchors.block(start_row, 0, anchor_Cyclist.rows(), box_coder_size) = std::move(anchor_Cyclist);
    start_row += anchor_Cyclist.rows();
    anchors.block(start_row, 0, anchor_Van.rows(), box_coder_size) = std::move(anchor_Van);
    start_row += anchor_Van.rows();
    return anchors;
}