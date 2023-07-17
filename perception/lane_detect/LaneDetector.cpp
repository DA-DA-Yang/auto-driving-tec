#include "lane_detect.h"
#include <boost/filesystem.hpp>
#include <thread>

void procImage(std::string img_path)
{
    std::cout << "图像文件：" << img_path << std::endl;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    LaneDetect lane_detector;
    if (lane_detector.isInit() == false)
    {
        lane_detector.init(img.cols, img.rows, 460);
    }
    lane_detector.processImage(img);
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/Final_img.jpg", img);
    cv::namedWindow("Lane Detect", cv::WINDOW_AUTOSIZE);
    cv::imshow("Lane Detect", img);
    cv::waitKey();
}

void procVideo(std::string video_path)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        printf("could not read this video file...\n");
        return;
    }
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
                          (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = cap.get(cv::CAP_PROP_FOURCC);
    printf("current fps : %d \n", fps);
    // cv::VideoWriter writer("/auto-driving-tec/data/lane_detect/result/lane_detect.mp4",
    //                        fourcc, fps, S, true);
    cv::VideoWriter writer("/auto-driving-tec/data/lane_detect/result/lane_detect.mp4",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, S, true);
    cv::Mat frame;
    LaneDetect lane_detector;

    cv::namedWindow("Lane Detect", cv::WINDOW_AUTOSIZE);
    int count{};
    while (cap.read(frame))
    {
        // if (count == 552)
        // {
        //     std::cout << "test" << std::endl;
        //     std::string img_name = "/auto-driving-tec/data/lane_detect/frame_" + std::to_string(count) + ".jpg";
        //     cv::imwrite(img_name, frame);
        // }
        // if(count>=500)
        //     break;
        std::cout << std::endl
                  << "Frame = " << count << std::endl;
        if (lane_detector.isInit() == false)
        {
            lane_detector.init(frame.cols, frame.rows, 460);
        }
        lane_detector.processImage(frame);

        count++;
        if (writer.isOpened())
        {
            writer << frame;
            if (count >= 100)
                writer.release();
        }

        cv::imshow("Lane Detect", frame);

        cv::waitKey(40);
    }
    cap.release();
    writer.release();
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int main(int argc, char **argv)
{
    boost::filesystem::path file{argv[1]};
    std::string file_path(argv[1]);
    if (file.extension().string() == ".mp4")
    {
        std::thread proc_thread(procVideo, file_path);
        proc_thread.join();
    }
    if (file.extension().string() == ".jpg")
    {
        std::thread proc_thread(procImage, file_path);
        proc_thread.join();
    }

    return 0;
}