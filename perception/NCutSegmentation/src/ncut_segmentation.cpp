#include "ncut_segmentation.h"
#include "perception/NCutSegmentation/common/ncut_flags.h"

NCutSegmentation::NCutSegmentation(/* args */)
{
#ifdef DEBUG
    _pcl_viewer = pcl::visualization::PCLVisualizer::Ptr(
        new pcl::visualization::PCLVisualizer("3D Viewer"));
    _pcl_viewer->setBackgroundColor(0, 0, 0);
    //_pcl_viewer->addCoordinateSystem(1.0);
    //_pcl_viewer->initCameraParameters();
    _point_cloud_ptr = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    _rgb_cloud_ptr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
#endif
}

NCutSegmentation::~NCutSegmentation()
{
}

void NCutSegmentation::getSegments(std::vector<PointXYZI> &pointcloud)
{
    int num_points = static_cast<int>(pointcloud.size());
    std::vector<PointXYZI> pc;
    pc.resize(num_points);
    memcpy(pc.data(), pointcloud.data(), num_points*3*sizeof(float));

    // 地面检测
    PointXYZ cloud_center{0.f, 0.f, 0.f};
    std::vector<int> non_ground_point_indixes;
    GroundDetector ground_detector;
    ground_detector.detect(cloud_center, pc, non_ground_point_indixes);
    std::vector<PointXYZI> pc_non_ground;
    int num_non_ground_poins = static_cast<int>(non_ground_point_indixes.size());
    for(int i=0;i<num_non_ground_poins;++i)
    {
        PointXYZI pt =pc[non_ground_point_indixes[i]];
        pc_non_ground.push_back(pt);
    }

    #ifdef DEBUG
    // 显示原始点云
    visualizePointCloud(pointcloud);
    // 显示去除地面后的点云
    visualizePointCloud(pc_non_ground);
    #endif

    float grid_radius=80.f;
    float cell_size = FLAGS_large_cut_size;
    // 执行floodfill
    FloodFill ff(grid_radius, cell_size);
    std::vector<std::vector<int>> point_index_in_segments;
    std::vector<int> num_cells_per_segment;
    ff.getSegments(pc_non_ground, point_index_in_segments, num_cells_per_segment);

    #ifdef DEBUG
    // 显示经过floodfill后的连通域
    visualizeSegments(pc_non_ground, point_index_in_segments);
    #endif

    // 执行ncut
    std::vector<std::vector<PointXYZ>> polygon;
    std::vector<std::vector<PointXYZ>> polygon_maxZ;
    std::vector<BOX_PCL> boxes;
    NCut ncut;
    int num_ff_segments = static_cast<int>(point_index_in_segments.size());
    // 对经过floodfill处理的连通域进行ncut处理
    for(int i = 0; i < num_ff_segments; ++i)
    {
        std::vector<PointXYZI> pc_segment;
        int num_points_in_segment = static_cast<int>(point_index_in_segments[i].size());
        for(int j = 0; j < num_points_in_segment; ++j)
        {
            pc_segment.push_back(pc_non_ground[point_index_in_segments[i][j]]);
        }
        // ncut
        ncut.segments(&pc_segment);
        // 获取分割结果
        // #ifdef DEBUG
        //         visualizeSegments(pc_segment, ncut.getPointIndicesInSegments());
        // #endif
        // 对经过ncut处理的每一个连通域，进行检测框计算
        int num_ncut_segments = ncut.getNumSegments();
        for(int id_seg = 0;id_seg < num_ncut_segments; ++id_seg)
        {
            int num_points_in_ncut_seg = static_cast<int>(ncut.getPointIndicesInSegments()[id_seg].size());
            // 点很少连通域中忽略不计
            if(num_points_in_ncut_seg < _num_points_threshold)
                continue;

            std::vector<PointXYZI> pc_in_ncut_seg;
            for(int id_p =0;id_p<num_points_in_ncut_seg;++id_p)
            {
                pc_in_ncut_seg.push_back(pc_segment[ncut.getPointIndicesInSegments()[id_seg][id_p]]);
            }
            BoundingBox bbox_detector;
            BOX_PCL box;
            if(bbox_detector.buildPolygon(pc_in_ncut_seg))
            {
                auto cloud_center = bbox_detector.getCloudCenter();
                auto box_size = bbox_detector.getBboxSize();
                auto angle_yaw = bbox_detector.getAngleYaw();
                box.x = cloud_center(0);
                box.y = cloud_center(1);
                box.z = cloud_center(2);
                box.l = box_size(0);
                box.w = box_size(1);
                box.h = box_size(2);
                box.r = angle_yaw;
                box.n = 0;
                box.label = NUMBER_LABEL_MAP.at(box.n);
                box.color = pcl::RGB{255, 255, 0};
                boxes.push_back(box);

                polygon.push_back(bbox_detector.getPolygon());
                polygon_maxZ.push_back(bbox_detector.getPolygon());
                int np = static_cast<int>(polygon_maxZ.back().size());
                for (int i = 0; i < np; ++i)
                {
                    polygon_maxZ.back()[i].z += box.h;
                }
            }
            // #ifdef DEBUG
            // std::vector<BOX_PCL> boxes_tmp;
            // boxes_tmp.push_back(box);
            // visualizeBboxes(pc_in_ncut_seg, boxes_tmp);
            // #endif
        }
        // #ifdef DEBUG
        // std::cout << "Num of bboxes: " << boxes.size() << std::endl;
        // visualizeBboxes(pc_non_ground, boxes);
        // #endif
    }
    #ifdef DEBUG
    std::cout << "Num of bboxes: " << boxes.size() << std::endl;
    visualizePolygon(pointcloud, polygon, polygon_maxZ);
    visualizePolygonAndBboxes(pointcloud, polygon, boxes);
#endif
}

#ifdef DEBUG
void NCutSegmentation::visualizePointCloud(const std::vector<PointXYZI> &pointcloud)
{
    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _point_cloud_ptr->clear();

    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZI pt;
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
        pt.intensity = pointcloud[i].i;
        _point_cloud_ptr->push_back(pt);
    }
    // 按"intensity"着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> field_color(_point_cloud_ptr, "intensity");
    _pcl_viewer->addPointCloud<pcl::PointXYZI>(_point_cloud_ptr, field_color, "point cloud");
    _pcl_viewer->spin();
}

void NCutSegmentation::visualizeSegments(const std::vector<PointXYZI> &pointcloud, 
                                         const std::vector<std::vector<int>>& point_index_in_segments)
{
    //pointcloud: 点云
    //point_index_in_segments: 每个连通域中的点的索引

    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _rgb_cloud_ptr->clear(); 
    
    unsigned int seed;
    int num_segments = static_cast<int>(point_index_in_segments.size());
    for (int i = 0; i < num_segments; ++i)
    {
        // 每个连通域赋予不同的颜色
        int red = 50 + rand_r(&seed) % 206;
        int green = 50 + rand_r(&seed) % 206;
        int blue = 50 + rand_r(&seed) % 206;
        int num_points_per_segment = static_cast<int>(point_index_in_segments[i].size());
        for (int j = 0; j < num_points_per_segment; ++j)
        {
            pcl::PointXYZRGB pt;
            pt.x = pointcloud[point_index_in_segments[i][j]].x;
            pt.y = pointcloud[point_index_in_segments[i][j]].y;
            pt.z = pointcloud[point_index_in_segments[i][j]].z;
            pt.r = red;
            pt.g = green;
            pt.b = blue;
            _rgb_cloud_ptr->push_back(pt);
        }
    } 
    _pcl_viewer->addPointCloud(_rgb_cloud_ptr, "segments");
    _pcl_viewer->spin(); 
}

void NCutSegmentation::visualizePolygon(const std::vector<PointXYZI> &pointcloud, const std::vector<std::vector<PointXYZ>> &polygon)
{
    // pointcloud: 点云
    // polygon: 包围区域的多边形凸点
    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _point_cloud_ptr->clear();

    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZI pt;
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
        pt.intensity = pointcloud[i].i;
        _point_cloud_ptr->push_back(pt);
    }
    // 按"intensity"着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> field_color(_point_cloud_ptr, "intensity");
    _pcl_viewer->addPointCloud<pcl::PointXYZI>(_point_cloud_ptr, field_color, "point cloud");

    //===添加polygon============================================
    int num_polygon = static_cast<int>(polygon.size());
    _polygon.clear();
    _polygon.resize(num_polygon);
    
    for (int i = 0; i < num_polygon; i++)
    {
        _polygon[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
         int num_points_in_polygon = static_cast<int>(polygon[i].size());
         _polygon[i]->reserve(num_points_in_polygon);
         for(int j=0;j<num_points_in_polygon;++j)
         {  
            pcl::PointXYZ pt;
            pt.x=polygon[i][j].x;
            pt.y=polygon[i][j].y;
            pt.z=polygon[i][j].z;
            _polygon[i]->push_back(pt);
         }
        _pcl_viewer->addPolygon<pcl::PointXYZ>(_polygon[i],255,255,0, std::to_string(i)); // 显示为黄色
    }
    _pcl_viewer->spin();
}

void NCutSegmentation::visualizePolygon(const std::vector<PointXYZI> &pointcloud, const std::vector<std::vector<PointXYZ>> &polygon_minZ, const std::vector<std::vector<PointXYZ>> &polygon_maxZ)
{
    if (polygon_minZ.size() != polygon_maxZ.size())
        return;
    // pointcloud: 点云
    // polygon: 包围区域的多边形凸点
    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _point_cloud_ptr->clear();

    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZI pt;
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
        pt.intensity = pointcloud[i].i;
        _point_cloud_ptr->push_back(pt);
    }
    // 按"intensity"着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> field_color(_point_cloud_ptr, "intensity");
    _pcl_viewer->addPointCloud<pcl::PointXYZI>(_point_cloud_ptr, field_color, "point cloud");

    //===添加polygon============================================
    int num_polygon = static_cast<int>(polygon_minZ.size());
    _polygon.clear();
    _polygon.resize(num_polygon);
    _polygon_maxZ.clear();
    _polygon_maxZ.resize(num_polygon);
    int count = 0;
    for (int i = 0; i < num_polygon; ++i)
    {
        _polygon[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        _polygon_maxZ[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        int num_points_in_polygon = static_cast<int>(polygon_minZ[i].size());
        _polygon[i]->reserve(num_points_in_polygon);
        _polygon_maxZ[i]->reserve(num_points_in_polygon);
        for (int j = 0; j < num_points_in_polygon; ++j)
        {
            pcl::PointXYZ pt;
            pt.x = polygon_minZ[i][j].x;
            pt.y = polygon_minZ[i][j].y;
            pt.z = polygon_minZ[i][j].z;
            _polygon[i]->push_back(pt);
            pcl::PointXYZ pt_maxZ;
            pt_maxZ.x = polygon_maxZ[i][j].x;
            pt_maxZ.y = polygon_maxZ[i][j].y;
            pt_maxZ.z = polygon_maxZ[i][j].z;
            _polygon_maxZ[i]->push_back(pt_maxZ);
            _pcl_viewer->addLine<pcl::PointXYZ, pcl::PointXYZ>(_polygon[i]->at(j), _polygon_maxZ[i]->at(j), 255, 255, 0, "line" + std::to_string(count));
            count++;
        }
        _pcl_viewer->addPolygon<pcl::PointXYZ>(_polygon[i], 255, 255, 0, "polygon" + std::to_string(i));          // 显示为黄色
        _pcl_viewer->addPolygon<pcl::PointXYZ>(_polygon_maxZ[i], 255, 255, 0, "polygon_max" + std::to_string(i)); // 显示为黄色
    }
    _pcl_viewer->spin();
}

void NCutSegmentation::visualizeBboxes(const std::vector<PointXYZI> &pointcloud, const std::vector<BOX_PCL> &bboxes)
{
    // pointcloud: 点云
    // bboxes: 每个连通域中的bbox

    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _point_cloud_ptr->clear();

    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZI pt;
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
        pt.intensity = pointcloud[i].i;
        _point_cloud_ptr->push_back(pt);
    }
    // 按"intensity"着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> field_color(_point_cloud_ptr, "intensity");
    _pcl_viewer->addPointCloud<pcl::PointXYZI>(_point_cloud_ptr, field_color, "point cloud");

    //===添加3D-boxes============================================
    int num_bboxes = static_cast<int>(bboxes.size());

    for (int idx = 0; idx < num_bboxes; idx++)
    {
        // 绕z轴旋转的角度调整
        Eigen::AngleAxisf rotation_vector(bboxes[idx].r, Eigen::Vector3f(0, 0, 1));
        // 绘制对象检测框，参数为三维坐标，长宽高还有旋转角度以及长方体名称
        std::string label_name = bboxes[idx].label + "-" + std::to_string(idx);
        _pcl_viewer->addCube(Eigen::Vector3f(bboxes[idx].x, bboxes[idx].y, bboxes[idx].z),
                             Eigen::Quaternionf(rotation_vector), bboxes[idx].l, bboxes[idx].w, bboxes[idx].h, label_name);
        // 设置检测框只有骨架
        _pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, label_name);
        // 设置检测框的颜色属性
        _pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, bboxes[idx].color.r, bboxes[idx].color.g, bboxes[idx].color.b, label_name);
    }
    _pcl_viewer->spin();
}

void NCutSegmentation::visualizePolygonAndBboxes(const std::vector<PointXYZI> &pointcloud, const std::vector<std::vector<PointXYZ>> &polygon, const std::vector<BOX_PCL> &bboxes)
{
    // pointcloud: 点云
    // bboxes: 每个连通域中的bbox

    _pcl_viewer->removeAllPointClouds(0);
    _pcl_viewer->removeAllShapes(0);
    _point_cloud_ptr->clear();

    int num_points = static_cast<int>(pointcloud.size());
    for (int i = 0; i < num_points; ++i)
    {
        pcl::PointXYZI pt;
        pt.x = pointcloud[i].x;
        pt.y = pointcloud[i].y;
        pt.z = pointcloud[i].z;
        pt.intensity = pointcloud[i].i;
        _point_cloud_ptr->push_back(pt);
    }
    // 按"intensity"着色
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> field_color(_point_cloud_ptr, "intensity");
    _pcl_viewer->addPointCloud<pcl::PointXYZI>(_point_cloud_ptr, field_color, "point cloud");

    //===添加polygon============================================
    int num_polygon = static_cast<int>(polygon.size());
    _polygon.clear();
    _polygon.resize(num_polygon);

    for (int i = 0; i < num_polygon; i++)
    {
        _polygon[i] = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        int num_points_in_polygon = static_cast<int>(polygon[i].size());
        _polygon[i]->reserve(num_points_in_polygon);
        for (int j = 0; j < num_points_in_polygon; ++j)
        {
            pcl::PointXYZ pt;
            pt.x = polygon[i][j].x;
            pt.y = polygon[i][j].y;
            pt.z = polygon[i][j].z;
            _polygon[i]->push_back(pt);
        }
        _pcl_viewer->addPolygon<pcl::PointXYZ>(_polygon[i], 255, 255, 0, std::to_string(i)); // 显示为黄色
    }

    //===添加3D-boxes============================================
    int num_bboxes = static_cast<int>(bboxes.size());

    for (int idx = 0; idx < num_bboxes; idx++)
    {
        // 绕z轴旋转的角度调整
        Eigen::AngleAxisf rotation_vector(bboxes[idx].r, Eigen::Vector3f(0, 0, 1));
        // 绘制对象检测框，参数为三维坐标，长宽高还有旋转角度以及长方体名称
        std::string label_name = bboxes[idx].label + "-" + std::to_string(idx);
        _pcl_viewer->addCube(Eigen::Vector3f(bboxes[idx].x, bboxes[idx].y, bboxes[idx].z),
                             Eigen::Quaternionf(rotation_vector), bboxes[idx].l, bboxes[idx].w, bboxes[idx].h, label_name);
        // 设置检测框只有骨架
        _pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, label_name);
        // 设置检测框的颜色属性
        _pcl_viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, bboxes[idx].color.r, bboxes[idx].color.g, bboxes[idx].color.b, label_name);
    }
    _pcl_viewer->spin();
}

#endif
