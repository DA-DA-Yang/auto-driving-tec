#include "pfe_process.h"
#include "pp_base.h"
#include "voxel.h"
#include "pointpillarsnet.h"

pfe_process::pfe_process(/* args */)
{
    float_buf_.reset(new float[input_c_ * input_h_ * input_w_]);
}

pfe_process::~pfe_process()
{
}

void pfe_process::process(void *ptr, int size)
{
    /*加载点云*/
    auto ptr_tmp = (PointXYZI *)ptr;
    MatrixXf res(4, size);
    auto stride = sizeof(PointXYZI);
    for (int i = 0; i < size; i++)
    {
        auto &data = ptr_tmp[i];
        PointXYZI pre_point;
        pre_point.x = data.x;
        pre_point.y = data.y;
        pre_point.z = data.z;
        pre_point.intensity = data.intensity;
        memcpy((char *)res.data() + i * stride, &pre_point.x, 4);
        memcpy((char *)res.data() + i * stride + 4, &pre_point.y, 4);
        memcpy((char *)res.data() + i * stride + 8, &pre_point.z, 4);
        memcpy((char *)res.data() + i * stride + 12, &pre_point.intensity, 4);
    }
    // 预处理
    auto points = std::move(res.transpose());
    // 体素化
    Voxel voxel = Voxel();
    PointsInVoxels piv(std::move(voxel.points_to_voxels(points))); //(D, P, N)
    printf("pillar count:%i\n", (int)(piv.voxels.size()));
    // 执行pointpillars网络，生成伪图像
    PointPillarsNet ppn;
    vector<float> batch_image = ppn.extract(piv); //(C, P)
    auto bsize = batch_image.size();
    printf("batch_image size: %d\n", (int)bsize);
    float *p_npy_data_temp = &batch_image[0];
    float *p_npy_data = float_buf_.get();
    {
        for (int c = 0; c < input_c_; c++)
        {
            for (int h = 0; h < input_h_; h++)
            {
                for (int w = 0; w < input_w_; w++)
                {
                    *(p_npy_data + c * input_h_ * input_w_ + h * input_w_ + w) =
                        *(p_npy_data_temp + c * input_h_ * input_w_ + h * input_w_ + w);
                }
            }
        }
    }
}