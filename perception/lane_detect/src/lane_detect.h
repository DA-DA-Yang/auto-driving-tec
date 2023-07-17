#pragma once

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30
#define K_THRESHOLD 0.05
#define B_THRESHOLD 5
#define NUM_FRAME 10
#define MIN_DISTANCE 20
#define MAX_DISTANCE 40
#define CURVE_NUM_POINTS 10
#define CURVE_NUM_GROUPS 30
#define TO_DEGREE 0.017453292519
enum
{
    SCAN_STEP = 2,            // in pixels
    LINE_REJECT_DEGREES = 10, // in degrees
    BW_TRESHOLD = 250,        // edge response strength to recognize for 'WHITE'
    BORDERX = 10,             // px, skip this much from left & right borders
    MAX_RESPONSE_DIST = 5,    // px

    CANNY_MIN_TRESHOLD = 1,   // edge detector minimum hysteresis threshold
    CANNY_MAX_TRESHOLD = 100, // edge detector maximum hysteresis threshold

    HOUGH_TRESHOLD = 50,        // line approval vote threshold
    HOUGH_MIN_LINE_LENGTH = 50, // remove lines shorter than this treshold
    HOUGH_MAX_LINE_GAP = 100,   // join lines to one with smaller than this gaps
};

struct Lane
{
    Lane() {}
    Lane(cv::Point2i p0, cv::Point2i p1, float angle, float kl, float bl)
        : point_start(p0), point_end(p1), angle(angle), votes(0), visited(false), found(false), k(kl), b(bl) {}
    cv::Point2i point_start, point_end;
    float angle;
    int votes;
    bool visited, found;
    float k, b;                   // y=k*x+b;
    float a0{}, a1{}, a2{}, a3{}; // y=a0+a1*x+a2*x^2+a3*x^3;
};

class ExpMovingAverage
{
private:
    double alpha; // [0;1] less = more stable, more = less stable
    double oldValue;
    bool unset;

public:
    ExpMovingAverage()
        : alpha(0.3), unset(true)
    {
    }

    void clear()
    {
        unset = true;
    }

    void add(double value)
    {
        if (unset)
        {
            oldValue = value;
            unset = false;
        }
        double newValue = oldValue + alpha * (value - oldValue);
        // double newValue = value;
        oldValue = newValue;
    }

    double get()
    {
        return oldValue;
    }
};

struct Status
{
    Status() : reset(true), lost(0) {}
    ExpMovingAverage k, b, a0, a1, a2, a3{};
    bool reset;
    int lost;
};

struct FourCorners // 结构体定义，126行-134行，four_corners_t是一个变量
{
    cv::Point2f left_top;
    cv::Point2f left_bottom; // point2f代表2维，需要X,Y轴来确定
    cv::Point2f right_top;
    cv::Point2f right_bottom;
};

class LaneDetect
{
private:
    /* data */
public:
    LaneDetect(/* args */);
    ~LaneDetect();
    void init(int img_width, int img_height, int y_start);
    bool processImage(cv::Mat &image);
    bool processLanes(std::vector<cv::Vec4i> &lines, cv::Mat &edge_img, cv::Mat &roi_img, cv::Mat &origin_image);
    bool processSide(std::vector<Lane> &side_lines, cv::Mat &edge_img, bool is_right, cv::Mat &origin_image);
    void linesAverage(std::vector<Lane> &lines, cv::Mat &origin_image);
    bool findResponses(cv::Mat &edge_img, int begin_x, int end_x, int y, std::vector<int> &rsp);
    void setROI(cv::Mat &input, std::vector<cv::Point2i> &roi);
    void seclectWhite(cv::Mat &input, cv::Mat &output, int threshold = 220);
    void seclectYellow(cv::Mat &input, cv::Mat &output, int threshold = 180);
    float computeDistance(cv::Point2i line_p0, cv::Point2i line_p1, cv::Point2i point);
    float computeDistance(float k, float b, cv::Point2i point);

    void slideWindow(cv::Mat &edge_img, cv::Point2i point, float k, float b, std::vector<cv::Point2f> &points_line);
    void addPoints(const cv::Mat &image, const std::vector<cv::Point2f> &target_points, float k, float b,
                   std::vector<cv::Point2f> &points_line, int &y_min);
    void fitCurve(std::vector<double> &x, std::vector<double> &y, std::vector<double> &curve_params);
    void fitCurve(cv::Mat &image, const std::vector<cv::Point2f> &points, std::vector<double> &curve_params);
    void drawCurve(cv::Mat &image, const std::vector<double> &curve_params, int offset_y, int x_start, int x_end, int y_start, int y_end, std::vector<cv::Point2f> &curve_points);
    void drawCurve(cv::Mat &image, const std::vector<cv::Point2f> &line_points, int offset_y);
    void fillCurvePoly(cv::Mat &image, const std::vector<cv::Point2f> &left_curve_points, const std::vector<cv::Point2f> &right_curve_points);
    void drawPoints(cv::Mat &image, const std::vector<cv::Point2f> &points, int offset_x, int offset_y, cv::Scalar color);
    double getBezierValue(double x, const std::vector<double> &curve_params);

    void surfMatch(cv::Mat &cur_img, cv::Mat &last_img, cv::Mat &H_matrix);
    void CalcCorners(const cv::Mat &H, const cv::Mat &src, FourCorners &corners);
    void pointsTransform(std::vector<std::vector<cv::Point2f>> &frames_points, const cv::Mat &H_matrix, int offset_y,
                         std::vector<cv::Point2f> &target_points);
    inline bool isInit() { return inited_flag_; }

    std::vector<cv::Point2f> left_points_, right_points_;
    Status laneR, laneL;
    std::vector<cv::Point2i> ROI{cv::Point2i{540, 450}, cv::Point2i{740, 450},
                                 cv::Point2i{1230, 700}, cv::Point2i{100, 700}};

    std::vector<cv::KeyPoint> response_points_;
    cv::Mat m_Bezier3_;
    cv::Mat m_Bezier3_inv_;

    cv::Mat h_matrix_;
    std::vector<std::vector<cv::Point2f>> left_frames_points_, right_frames_points_;
    std::vector<cv::Point2f> left_last_points_, right_last_points_;
    std::vector<cv::Point2f> left_curve_points_, right_curve_points_;
    cv::Mat last_frame_;
    int frame_count_ = 0;
    int left_y_min_ = 0;
    int right_y_min_ = 0;
    int offset_y_;
    int img_width_;
    int img_height_;
    int y_start_;
    bool inited_flag_ = false;
};
