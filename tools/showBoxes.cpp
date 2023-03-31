
#include <fstream>
#include <iostream>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>

#include "base.h"
#include "define.h"

unsigned int g_index{};
bool g_contiued{false};

std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> loadpt(std::vector<boost::filesystem::path> stream)
{
    auto streamIterator = stream.begin();
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> array;
    while (streamIterator != stream.end())
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>);
        std::string ss((*streamIterator).string());
        std::cout << ss << std::endl;
        pcl::io::loadPCDFile(ss, *cloud1);
        array.push_back(cloud1);
        streamIterator++;
    }
    return array;
}

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer_void)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
    if (event.getKeySym() == "Down" && event.keyDown())
    {
        g_index++;
    }
    if (event.getKeySym() == "Up" && event.keyDown())
    {
        g_index--;
    }
    if (event.getKeySym() == "space" && event.keyDown())
    {
        // 连续播放
        g_contiued = g_contiued == false ? true : false;
    }
    if (event.getKeySym() == "r" && event.keyDown())
    {
        // 复位，回到初始状态
        g_index = 0;
        g_contiued = false;
    }
}

int main(int argc, char **argv)
{
    //[usage: ./ShowBoxes pcd_dir_path box_dir_path]
    if (argc <= 2)
    {
        std::cout << "ERROR: 参数不足，缺少文件夹路径！" << std::endl;
        return 0;
    }
    // 获取文件目录
    std::string pcd_dirPath(argv[1]);
    std::cout << "pcd path：" << argv[1] << std::endl;
    std::string box_dirPath(argv[2]);
    std::cout << "box path：" << argv[2] << std::endl;
    // 获取指定类型文件
    std::vector<boost::filesystem::path> stream = getFiles(pcd_dirPath, "pcd");

    // 从文件中加载点云
    // std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> array = loadpt(stream);

    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    // 注册键盘响应事件
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&viewer);

    // 创建点云指针
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

    // 显示点云
    if (!stream.empty())
    {
        int i_cur{}, i_last{-1};
        while (!viewer->wasStopped())
        {
            if (!g_contiued)
                i_cur = g_index;
            else
                i_cur = g_index++;
            //===添加点云================================================
            i_cur = i_cur >= stream.size() ? stream.size() - 1 : i_cur;
            i_cur = i_cur < 0 ? 0 : i_cur;
            if (i_cur == i_last)
            {
                viewer->spinOnce(100);
                continue;
            }
            // 加载点云
            cloud->clear();
            std::string ss(stream[i_cur].string());
            std::cout << "当前加载：" << ss << std::endl;
            pcl::io::loadPCDFile(ss, *cloud);

            viewer->removePointCloud("sample cloud");
            // 按"intensity"着色
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> fildColor(cloud, "intensity");
            // 添加点云
            viewer->addPointCloud<pcl::PointXYZI>(cloud, fildColor, "sample cloud");
            // 设置点云大小
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
            //===========================================================

            //===添加3D-boxes============================================
            viewer->removeAllShapes();
            std::vector<BOX_PCL> boxes;
            // 从文件中读取感知结果
            std::string txt_path = box_dirPath + stream[i_cur].stem().string() + ".txt";

            std::ifstream txt_file;
            std::stringstream txt_ss;
            txt_file.open(txt_path.data(), std::ios::in);
            if (txt_file.is_open())
            {
                std::string str_line;
                while (getline(txt_file, str_line))
                {
                    txt_ss.str(str_line);
                    std::string str_single;
                    std::vector<std::string> str_vec;
                    int i = 0;
                    // 按照空格分隔
                    while (getline(txt_ss, str_single, ' '))
                    {
                        str_vec.push_back(str_single);
                    }
                    // 必须加，不然写不到string里。
                    txt_ss.clear();
                    // 有8个参数才表明是感知结果
                    if (str_vec.size() == 8)
                    {
                        BOX_PCL box;
                        box.x = atof(str_vec[0].c_str());
                        box.y = atof(str_vec[1].c_str());
                        box.z = atof(str_vec[2].c_str());
                        box.l = atof(str_vec[3].c_str());
                        box.w = atof(str_vec[4].c_str());
                        box.h = atof(str_vec[5].c_str());
                        box.r = atof(str_vec[6].c_str());
                        box.n = atoi(str_vec[7].c_str());
                        box.label = NUMBER_LABEL_MAP.at(box.n);
                        box.color = LABEL_COLOR_MAP.at(box.label);
                        boxes.push_back(box);
                    }
                }
                txt_file.close();
            }
            for (size_t idx = 0, idx_max = boxes.size(); idx < idx_max; idx++)
            {
                // 绕z轴旋转的角度调整
                Eigen::AngleAxisf rotation_vector(-boxes[idx].r, Eigen::Vector3f(0, 0, 1));
                // 绘制对象检测框，参数为三维坐标，长宽高还有旋转角度以及长方体名称
                std::string label_name = boxes[idx].label + "-" + std::to_string(idx);
                viewer->addCube(Eigen::Vector3f(boxes[idx].x, boxes[idx].y, boxes[idx].z),
                                Eigen::Quaternionf(rotation_vector), boxes[idx].l, boxes[idx].w, boxes[idx].h, label_name);
                // 设置检测框只有骨架
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, label_name);
                // 设置检测框的颜色属性
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, boxes[idx].color.r, boxes[idx].color.g, boxes[idx].color.b, label_name);
            }
            //===========================================================
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::milliseconds(100));
            i_last = i_cur;
        };
    }

    return 0;
}
