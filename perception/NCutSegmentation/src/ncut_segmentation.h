#ifdef DEBUG
#include "pcl/visualization/pcl_visualizer.h"
#endif
#include "define.h"
#include "ground_detector.h"
#include "floodfill.h"
#include "ncut.h"
#include "perception/common/bbox/bounding_box.h"
class NCutSegmentation
{

public:
    NCutSegmentation(/* args */);
    ~NCutSegmentation();

    void getSegments(std::vector<PointXYZI> &pointcloud);

private:
    int _num_points_threshold{10}; // 小于10个点的连通域不再处理

#ifdef DEBUG
    // 可视化函数
    pcl::visualization::PCLVisualizer::Ptr _pcl_viewer;
    pcl::PointCloud<pcl::PointXYZI>::Ptr _point_cloud_ptr;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr _rgb_cloud_ptr;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> _polygon; 
    void visualizePointCloud(const std::vector<PointXYZI> &pointcloud);
    void visualizeSegments(const std::vector<PointXYZI> &pointcloud, const std::vector<std::vector<int>>& point_index_in_segments);
    void visualizePolygon(const std::vector<PointXYZI> &pointcloud, const std::vector<std::vector<PointXYZ>> &polygon);
    void visualizeBboxes(const std::vector<PointXYZI> &pointcloud, const std::vector<BOX_PCL> &bboxes);

#endif
};


