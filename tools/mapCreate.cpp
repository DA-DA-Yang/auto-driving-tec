#include <opencv4/opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <dirent.h>

using namespace cv;
using namespace std;

struct Image_Pos
{
    int x;
    int y;
    string path;
};

// 读取栅格图片的路径与坐标
bool readFileList(string dirPath, vector<Image_Pos> &files, int &min_x, int &max_x, int &min_y, int &max_y);

int main(int argc, char **argv)
{
    // 读取参数
    if (argc <= 2)
    {
        cout << "error: no dirpath input or output!\n";

        return 0;
    }
    cout << "---Dir path:\n"
         << argv[1] << "\n";
    string inDirPath{argv[1]};
    string outDirPath{argv[2]};

    // 读取图片文件夹
    vector<Image_Pos> imgPaths;
    // 栅格图片的最大/最小的x、y坐标
    int min_x, max_x, min_y, max_y;
    if (!readFileList(inDirPath, imgPaths, min_x, max_x, min_y, max_y))
        return 0;
    if (imgPaths.size() == 0)
    {
        cout << "No imgs in Dir\n";
        return 0;
    }

    // 确定合并后的图像大小
    Mat img = imread(imgPaths[0].path, 0);
    int height = img.rows;
    int width = img.cols;
    int nw = max_x - min_x + 1;
    int nh = max_y - min_y + 1;
    Mat res = Mat::zeros(height * nh, width * nw, img.type());

    // 读取所有图片并完成合并
    for (int i = 0; i < imgPaths.size(); ++i)
    {
        Mat img_t = imread(imgPaths[i].path, 0);
        if (img_t.rows != height || img.cols != width)
        {
            std::cout << "error in " << imgPaths[i].path << " Image size not match!";
            continue;
        }
        int px = imgPaths[i].x - min_x + 1;
        int py = imgPaths[i].y - min_y + 1;
        img_t.copyTo(res(Rect((px - 1) * width, (py - 1) * height, width, height)));
    }

    // 竖直方向翻转
    Mat res_flip;
    flip(res, res_flip, 0); // 0为x轴翻转，>0为y轴翻转，<0为双轴翻转

    // 防止图像过大，进行缩放
    double mpp = 0.125;
    double distance = 1024;
    double xorigin = (double(max_x - min_x + 1) * 0.5 + (double)min_x) * (distance * mpp);
    double yorigin = (double(max_y - min_y + 1) * 0.5 + (double)min_y) * (distance * mpp);
    double scale = mpp;
    if (res_flip.cols < 5000 && res_flip.rows < 5000)
        scale = 1;
    Mat out;
    resize(res_flip, out, Size(res_flip.cols * scale, res_flip.rows * scale));

    // 保存地图
    string outImagePath = inDirPath + "background.jpg";
    imwrite(outImagePath.data(), out);

    // 保存地图信息
    string outPosPath = inDirPath + "background_pos.txt";
    ofstream outfile(outPosPath.data(), ios::trunc);
    outfile << "mpp: " << mpp << "\n"
            << fixed
            << "xres: " << res_flip.cols << "\n"
            << "yres: " << res_flip.rows << "\n"
            << "xorigin: " << (int)xorigin << "\n"
            << "yorigin: " << (int)yorigin << "\n";

    cout
        << "---map position:\n "
        << "mpp: " << mpp << "\n"
        << "xres: " << res_flip.cols << "\n"
        << "yres: " << res_flip.rows << "\n"
        << "xorigin: " << (int)xorigin << "\n"
        << "yorigin: " << (int)yorigin << "\n";

    std::cout
        << "========================================\n"
        << "Done!\n"
        << "========================================\n";
    return 1;
}

bool readFileList(string dirPath, vector<Image_Pos> &files, int &min_x, int &max_x, int &min_y, int &max_y)
{
    min_x = max_x = min_y = max_y = 0;
    DIR *dir;
    struct dirent *ptr;

    if ((dir = opendir(dirPath.c_str())) == NULL)
    {
        perror("Open dir error...");
        exit(1);
    }

    while ((ptr = readdir(dir)) != NULL)
    {
        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
            continue;
        else if (ptr->d_type == 4) // d_type=4为dir
        {
            int y = stoi(ptr->d_name);
            if (min_y == 0)
                min_y = y;
            else if (y < min_y)
                min_y = y;
            if (max_y == 0)
                max_y = y;
            else if (y > max_y)
                max_y = y;

            // 开启第二层文件夹
            string dirPath_2 = dirPath + "/" + ptr->d_name;
            DIR *dir_2;
            struct dirent *ptr_2;

            if ((dir_2 = opendir(dirPath_2.c_str())) == NULL)
            {
                perror("Open dir error...");
                exit(1);
            }
            while ((ptr_2 = readdir(dir_2)) != NULL)
            {
                if (strcmp(ptr_2->d_name, ".") == 0 || strcmp(ptr_2->d_name, "..") == 0)
                    continue;
                if (string(ptr_2->d_name).find_first_of(".") == 0)
                    continue;
                else if (ptr_2->d_type == 8) // d_type=8为file
                {
                    string a = ptr_2->d_name;
                    int pe = a.find_last_of(".");
                    string file_type = a.substr(pe + 1); // 文件后缀
                    string file_name = a.substr(0, pe);  // 文件前缀

                    if (file_type == "png") // 若想获取其他类型文件只需修改png为对应后缀
                    {

                        // 获得x坐标
                        int x = stoi(file_name);
                        if (min_x == 0)
                            min_x = x;
                        else if (x < min_x)
                            min_x = x;
                        if (max_x == 0)
                            max_x = x;
                        else if (x > max_x)
                            max_x = x;
                        string file_path = dirPath_2 + "/" + ptr_2->d_name;
                        Image_Pos img{x, y, file_path};
                        files.push_back(img);
                    }
                }
            }
        }
    }
    closedir(dir);
    if (min_x <= 0 || max_x <= 0 || min_y <= 0 || max_y <= 0)
    {
        files.clear();
        perror("Open dir error...");
        exit(1);
    }
    return 1;
}