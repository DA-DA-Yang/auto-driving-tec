#include "bounding_box.h"
#include "convex_hull_2d.h"

static const float gEpsilon = 1e-6f;
static const float gEpsilonForSize = 1e-2f;
static const float gEpsilonForLine = 1e-3f;

BoundingBox::BoundingBox(/* args */)
{
}

BoundingBox::~BoundingBox()
{
}

bool BoundingBox::getBbox(const std::vector<PointXYZI> &pointcloud, BOX_PCL &box)
{
    int num_points = static_cast<int>(pointcloud.size());
    if(num_points < 1)
        return false;
    // 创建点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->resize(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZ &pt = (*cloud)[i];
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
    }
    // 创建特征提取器
    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud);
    feature_extractor.compute();

    pcl::PointXYZ min_point_OBB;
    pcl::PointXYZ max_point_OBB;
    pcl::PointXYZ position_OBB;
    Eigen::Matrix3f rotational_matrix_OBB;
    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);

    // 输出结果
    box.x = position_OBB.x;
    box.y = position_OBB.y;
    box.z = position_OBB.z;
    box.l = max_point_OBB.x - min_point_OBB.x;
    box.w = max_point_OBB.y - min_point_OBB.y;
    box.h = max_point_OBB.z - min_point_OBB.z;
    box.r = atan2(rotational_matrix_OBB(1, 0), rotational_matrix_OBB(0, 0));

    return true;
}

bool BoundingBox::buildPolygon(const std::vector<PointXYZI> &pointcloud)
{
    int num_points = static_cast<int>(pointcloud.size());
    _pointcloud.resize(num_points);
    memcpy(_pointcloud.data(), pointcloud.data(), num_points * 4 * sizeof(float));

    // 计算凸包
    _computePolygon2D(_pointcloud);

    // 计算包围框的尺寸与中心
    _computePolygonSizeAndCenter(_polygon);

    return true;
}

void BoundingBox::_computePolygon2D(std::vector<PointXYZI> &pointcloud)
{
    _getMinMax3D(pointcloud, &_min_pt, &_max_pt);
    if (pointcloud.size() < 4u)
    {
        _setDefaultValue(_min_pt, _max_pt);
        return;
    }
    
    // 直线扰动
    _linePerturbation(pointcloud);

    // 计算凸包
    common::ConvexHull2D<std::vector<PointXYZI>, std::vector<PointXYZ>> hull;
    hull.GetConvexHull(pointcloud, &_polygon);
}

void BoundingBox::_getMinMax3D(const std::vector<PointXYZI> &pointcloud, Eigen::Vector3f *min_pt, Eigen::Vector3f *max_pt)
{
    // brief: 从三维点云中获取最小最大的坐标，并非是一个点

    (*min_pt)[0] = (*min_pt)[1] = (*min_pt)[2] = std::numeric_limits<float>::max();
    (*max_pt)[0] = (*max_pt)[1] = (*max_pt)[2] = -std::numeric_limits<float>::max();
    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        const PointXYZI &pt = pointcloud[i];
        if (std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z))
        {
            continue;
        }
        (*min_pt)[0] = std::min((*min_pt)[0], pt.x);
        (*max_pt)[0] = std::max((*max_pt)[0], pt.x);
        (*min_pt)[1] = std::min((*min_pt)[1], pt.y);
        (*max_pt)[1] = std::max((*max_pt)[1], pt.y);
        (*min_pt)[2] = std::min((*min_pt)[2], pt.z);
        (*max_pt)[2] = std::max((*max_pt)[2], pt.z);
    }
}

void BoundingBox::_setDefaultValue(Eigen::Vector3f &min_pt, Eigen::Vector3f &max_pt)
{
    // brief: 设置默认值

    for (int i = 0; i < 3; i++)
    {
        if (max_pt[i] - min_pt[i] < gEpsilonForSize)
        {
            max_pt[i] = max_pt[i] + gEpsilonForSize / 2;
            min_pt[i] = min_pt[i] - gEpsilonForSize / 2;
        }
    }

    _cloud_center(0) = (min_pt[0] + max_pt[0]) / 2;
    _cloud_center(1) = (min_pt[1] + max_pt[1]) / 2;
    _cloud_center(2) = (min_pt[2] + max_pt[2]) / 2;

    _box_size(0) = max_pt[0] - min_pt[0];
    _box_size(1) = max_pt[1] - min_pt[1];
    _box_size(2) = max_pt[2] - min_pt[2];

    _direction = Eigen::Vector3f(1.0, 0.0, 0.0);

    if(_polygon.size()<4)
        _polygon.resize(4);
    // 构建多边形
    _polygon[0].x = static_cast<double>(min_pt[0]);
    _polygon[0].y = static_cast<double>(min_pt[1]);
    _polygon[0].z = static_cast<double>(min_pt[2]);

    _polygon[1].x = static_cast<double>(max_pt[0]);
    _polygon[1].y = static_cast<double>(min_pt[1]);
    _polygon[1].z = static_cast<double>(min_pt[2]);

    _polygon[2].x = static_cast<double>(max_pt[0]);
    _polygon[2].y = static_cast<double>(max_pt[1]);
    _polygon[2].z = static_cast<double>(min_pt[2]);

    _polygon[3].x = static_cast<double>(min_pt[0]);
    _polygon[3].y = static_cast<double>(max_pt[1]);
    _polygon[3].z = static_cast<double>(min_pt[2]);
}

bool BoundingBox::_linePerturbation(std::vector<PointXYZI> &pointcloud)
{
    // @brief: decide whether input cloud is on the same line. if ture, add perturbation.

    int num_points = static_cast<int>(pointcloud.size());
    if (num_points >= 3)
    {
        int start_point = 0;
        int end_point = 1;
        float diff_x = pointcloud[start_point].x - pointcloud[end_point].x;
        float diff_y = pointcloud[start_point].y - pointcloud[end_point].y;
        int idx = 0;
        for (idx = 2; idx < num_points; ++idx)
        {
            float tdiff_x = pointcloud[idx].x - pointcloud[start_point].x;
            float tdiff_y = pointcloud[idx].y - pointcloud[start_point].y;
            if (fabs(diff_x * tdiff_y - tdiff_x * diff_y) > gEpsilonForLine)
            {
                // 返回false说明点云中的点不在一条直线上
                return false;
            }
        }
        // 如果所有点在一条直线上，则添加小扰动
        pointcloud[0].x += gEpsilonForLine;
        pointcloud[1].y += gEpsilonForLine;
        // 返回true说明点云中的点在一条直线上
        return true;
    }
    return true;
}

void BoundingBox::_computePolygonSizeAndCenter(std::vector<PointXYZ>& polygon)
{      
    int num_points = static_cast<int>(polygon.size());

    // 计算左右两个极端点
    Eigen::Vector3d max_point(polygon[0].x, polygon[0].y, polygon[0].z);
    Eigen::Vector3d min_point(polygon[0].x, polygon[0].y, polygon[0].z);
    int max_point_index = 0;
    int min_point_index = 0;
    for (int i = 1; i < num_points; ++i) 
    {
        Eigen::Vector3d p(polygon[i].x, polygon[i].y, polygon[i].z);
        Eigen::Vector3d ray = p;
        // clock direction，两个向量的叉乘，小于0，说明夹角大于180度
        // 也说明了该点在max_point的右边，则该点更大
        if (max_point[0] * ray[1] - ray[0] * max_point[1] < 0) 
        {
            max_point = ray;
            max_point_index = i;
        }
        // unclock direction
        if (min_point[0] * ray[1] - ray[0] * min_point[1] > 0) 
        {
            min_point = ray;
            min_point_index = i;
        }
    }

    if(max_point_index < min_point_index)
    {
        max_point_index = max_point_index + num_points;
    }

    double min_area = std::numeric_limits<double>::max();
    double best_length, best_width;
    Eigen::Vector3d best_center, best_direction;
    for (int i = min_point_index; i < max_point_index - 1; ++i)
    //for (int i = 0; i < num_points; ++i)
    {
        // 选择靠近lidar原点的两个点组成矩形的一边
        double length,width,area;
        Eigen::Vector3d center, direction;
        int p1_index = i % num_points;
        int p2_index = (i + 1) % num_points;
        _computeBboxAlongOneEdge(_polygon, p1_index, p2_index, &center, &length, &width, &area, &direction);
        if (min_area > area)
        {
            min_area = area;
            best_length = length;
            best_width = width;
            best_center = center;
            best_direction = direction;
        }
    }
    _cloud_center = best_center.cast<float>();
    _cloud_center(2) = (_max_pt(2) + _min_pt(2)) * 0.5;
    _box_size(0) = (float)best_length;
    _box_size(1) = (float)best_width;
    _box_size(2) = _max_pt(2) - _min_pt(2);
    _direction = best_direction.cast<float>();
    _angle_yaw = static_cast<float>(atan2(_direction(1), _direction(0)));
}

void BoundingBox::_computeBboxAlongOneEdge(std::vector<PointXYZ> &polygon, const int p1_index, const int p2_index,
                                           Eigen::Vector3d *center, double *length, double *width, double *area, Eigen::Vector3d *direction)
{
    // 把两个点组成矩形的一条边，求解polygon的最小外接矩形，同时计算中心点、长、宽、面积、方向

    int num_points = static_cast<int>(polygon.size());

    // 用于构成一条边的两个点，让点p2在p1的右侧
    Eigen::Vector3d p1, p2;
    if (polygon[p1_index].x <= polygon[p2_index].x)
    {
        p1 = Eigen::Vector3d(polygon[p1_index].x, polygon[p1_index].y, 0);
        p2 = Eigen::Vector3d(polygon[p2_index].x, polygon[p2_index].y, 0);
    }
    else
    {
        p1 = Eigen::Vector3d(polygon[p2_index].x, polygon[p2_index].y, 0);
        p2 = Eigen::Vector3d(polygon[p1_index].x, polygon[p1_index].y, 0);
    }
    
    Eigen::Vector3d p1_p2 = p2 - p1;
    double x2y2 = (p1_p2(0) * p1_p2(0) + p1_p2(1) * p1_p2(1));
    double norm_p1p2 = sqrt(x2y2);

    std::vector<Eigen::Vector3d> points_project;
    double min_width = std::numeric_limits<double>::max();
    double max_width = std::numeric_limits<double>::min();
    double max_height = std::numeric_limits<double>::min();
    Eigen::Vector3d max_height_point;
    Eigen::Vector3d max_height_point_project;
    Eigen::Vector3d max_width_point_project;
    Eigen::Vector3d min_width_point_project;

    // 遍历每个凸点，投影到边p1p2上
    for (int i = 0; i < num_points; ++i)
    {
        Eigen::Vector3d pt(polygon[i].x, polygon[i].y, 0);

        // 计算pt在直线p1p2上的投影
        // v2在v1上的投影：v1(v2.dot(v1))/(||v1||^2)=k(v1)
        // width_t为投影距离
        Eigen::Vector3d p1_pt = pt - p1;
        double k = p1_pt(0) * p1_p2(0) + p1_pt(1) * p1_p2(1);
        k /= x2y2;
        // 求投影点坐标
        Eigen::Vector3d pt_project;
        pt_project(0) = p1_p2(0) * k + p1(0);
        pt_project(1) = p1_p2(1) * k + p1(1);
        pt_project(2) = 0;
        // 求投影距离
        double width_t = k * norm_p1p2;
        // 叉乘表示两个向量组成四边形的面积
        double height_t = fabs(p1_p2(0) * p1_pt(1) - p1_p2(1) * p1_pt(0));
        // 求点到直线的距离：面积除以底为高
        height_t /= norm_p1p2;

        if (width_t < min_width)
        {
            // 在向量p1p2左侧的点，方向为负
            min_width = width_t;
            min_width_point_project = pt_project;
        }
        if (width_t > max_width)
        {
            // 在向量p1p2右侧的点，方向为正
            max_width = width_t;
            max_width_point_project = pt_project;
        }
        if (height_t > max_height)
        {
            // 距离最远的点
            max_height = height_t;
            max_height_point = pt;
            max_height_point_project = pt_project;
        }
    }

    // 求解box的长度与宽度
    if(min_width > 0 || max_width < 0)
    {
        *length = std::max(fabs(min_width), fabs(max_width));
    }
    else
    {
        *length = fabs(min_width) + fabs(max_width);
    }
    *width = fabs(max_height);

    // 求解box的中心坐标
    Eigen::Vector3d vp1 = min_width_point_project;
    Eigen::Vector3d vp2 = max_width_point_project;
    Eigen::Vector3d vp3 = max_height_point + max_width_point_project - max_height_point_project;
    Eigen::Vector3d vp4 = max_height_point + min_width_point_project - max_height_point_project;
    *center = (vp1 + vp2 + vp3 + vp4) * 0.25;
    
    // 求解方向
    *direction = p1_p2;
    // if (*length >= *width)
    // {
    //     *direction = p1_p2;
    // }
    // else
    // {
    //     double tmp = *length;
    //     *length = *width;
    //     *width = tmp;
    //     *direction = vp3-vp2;
    // }
    // 求解面积
    *area = (*length)*(*width);
}

void BoundingBox::_computePolygonSizeAndCenter()
{  
    if(_pointcloud.size()<4u)
        return;

    Eigen::Vector3f dir(1.f, 0.f, 0.f);
    _computeBboxSizeAndCenter2D(_pointcloud, dir, &_box_size, & _cloud_center);
    //_angle_yaw = static_cast<float>(atan2(dir(1), dir(0)));
    _angle_yaw = 0;
}

void BoundingBox::_computeBboxSizeAndCenter2D(const std::vector<PointXYZI> &pointcloud, const Eigen::Vector3f &dir, 
                                 Eigen::Vector3f *size, Eigen::Vector3f *center)
{
    // 建立平面投影矩阵
    Eigen::Matrix3d projection;
    Eigen::Vector3d dird((double)dir(0), (double)dir(1), 0.0);
    dird.normalize();
    projection << dird(0), dird(1), 0.0, -dird(1), dird(0), 0.0, 0.0, 0.0, 1.0;

    const double double_max = std::numeric_limits<double>::max();
    Eigen::Vector3d min_pt(double_max, double_max, double_max);
    Eigen::Vector3d max_pt(-double_max, -double_max, -double_max);
    Eigen::Vector3d loc_pt(0.0, 0.0, 0.0);
    int num_points = static_cast<int>(pointcloud.size());
    // 将点云投影在xy平面内
    for (int i = 0; i < num_points; ++i)
    {
        loc_pt = projection * Eigen::Vector3d(double(pointcloud[i].x), double(pointcloud[i].y), double(pointcloud[i].z));

        // 确定最小三维坐标
        min_pt(0) = std::min(min_pt(0), loc_pt(0));
        min_pt(1) = std::min(min_pt(1), loc_pt(1));
        min_pt(2) = std::min(min_pt(2), loc_pt(2));

        // 确定最大三维坐标
        max_pt(0) = std::max(max_pt(0), loc_pt(0));
        max_pt(1) = std::max(max_pt(1), loc_pt(1));
        max_pt(2) = std::max(max_pt(2), loc_pt(2));
    }

    // 确定包围框尺寸
    (*size) = (max_pt - min_pt).cast<float>();
    float minimum_size = std::numeric_limits<float>::epsilon();
    (*size)(0) = (*size)(0) <= minimum_size ? minimum_size : (*size)(0);
    (*size)(1) = (*size)(1) <= minimum_size ? minimum_size : (*size)(1);
    (*size)(2) = (*size)(2) <= minimum_size ? minimum_size : (*size)(2);

    // 确定投影后点云的中心，在原点云坐标系中的三维坐标
    Eigen::Vector3d coeff = (min_pt + max_pt) * 0.5;
    // coeff(2) = min_pt(2);
    (*center) = (projection.transpose() * coeff).cast<float>();    
}