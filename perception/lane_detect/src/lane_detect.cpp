#include "lane_detect.h"

LaneDetect::LaneDetect(/* args */)
{
    m_Bezier3_ = cv::Mat::zeros(cv::Size(4, 4), CV_64FC1);
    m_Bezier3_ = (cv::Mat_<double>(4, 4) << 1, -3, 3, -1, 0, 3, -6, 3, 0, 0, 3, -3, 0, 0, 0, 1);
    m_Bezier3_inv_ = m_Bezier3_.inv();
}

LaneDetect::~LaneDetect()
{
}

void LaneDetect::init(int img_width, int img_height, int y_start)
{
    img_width_ = img_width;
    img_height_ = img_height;
    offset_y_ = img_height / 2;
    for (size_t i = 0; i < ROI.size(); ++i)
    {
        ROI[i].y -= offset_y_;
        ROI[i].y = ROI[i].y < 0 ? 0 : ROI[i].y;
    }
    inited_flag_ = true;
    y_start_ = y_start;
}

bool LaneDetect::processImage(cv::Mat &image)
{
    // 复制一份原始图像
    frame_count_++;
    cv::Mat origin_img;
    image.copyTo(origin_img);

    if (origin_img.rows != img_height_ || origin_img.cols != img_width_)
    {
        printf("图像宽高不匹配！\n");
        return false;
    }

    // 获取图像的宽高
    int img_height = origin_img.rows;
    int img_width = origin_img.cols;
    cv::Size img_size{img_width, img_height / 2};

    // 创建用于算法处理的图像
    cv::Mat roi_img = cv::Mat(img_size, CV_8UC3, origin_img.type());
    cv::Mat gray_img = cv::Mat::zeros(img_size, CV_8U);
    cv::Mat edge_img = cv::Mat::zeros(img_size, CV_8U);

    // 裁剪一半图像，取原始图像的下半部分
    int offset = offset_y_;
    origin_img(cv::Rect(0, img_size.height, img_size.width, img_size.height)).convertTo(roi_img, roi_img.type());

#ifdef SHOW_DETAIL
    // // 显示中间线
    // cv::line(roi_img,
    //          cv::Point2f{float(img_width / 2), 0.f},
    //          cv::Point2f{float(img_width / 2), float(img_height / 2)},
    //          cv::Scalar(0, 255, 255), 1);
    // 保存ROI检测图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/roi_img.jpg", roi_img);
#endif

    // 灰度化
    cv::Mat gray_img_temp;
    cv::cvtColor(roi_img, gray_img_temp, cv::COLOR_BGR2GRAY);

    // 提取并叠加白色与黄色通道
    cv::Mat white, yellow;
    seclectWhite(roi_img, white);
    seclectYellow(roi_img, yellow);
    gray_img += white;
    gray_img += yellow;

#ifdef SHOW_DETAIL
    // 保存灰度图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/grey_img.jpg", gray_img);
#endif

#ifdef SHOW_DETAIL
    // 保存白色与黄色通道图像的二值化图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/white_img.jpg", white);
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/yellow_img.jpg", yellow);
    // 保存叠加图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/comp_img.jpg", gray_img);
#endif

    // canny边缘检测
    // 如何检测出道路边缘是至关重要的
    cv::Canny(gray_img, edge_img, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

#ifdef SHOW_DETAIL
    // 保存边缘图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/edge_img.jpg", edge_img);
#endif

    // 选择ROI，很重要，决定了边缘响应点的分布
    setROI(edge_img, ROI);

#ifdef SHOW_DETAIL
    // 保存ROI边缘图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/edgeROI_img.jpg", edge_img);
#endif

    // 霍夫变换
    std::vector<cv::Vec4i> hough_lines; // 储存霍夫变换的线
    cv::HoughLinesP(edge_img, hough_lines, 1, TO_DEGREE, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);

    // 利用灰度图像与车道阈值化图像叠加，检测车道边缘响应点
    gray_img += gray_img_temp;
    // 高斯滤波，滤波窗口影响边缘检测，不能太大，否则远端的边缘点无法检测，太小，则不够精确
    cv::GaussianBlur(gray_img, gray_img, cv::Size(11, 11), 0, 0);
    // canny边缘检测
    // 如何检测出道路边缘是至关重要的
    cv::Canny(gray_img, edge_img, CANNY_MIN_TRESHOLD, CANNY_MAX_TRESHOLD);

#ifdef SHOW_DETAIL
    // 保存边缘图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/edge2_img.jpg", edge_img);
#endif

    // 选择ROI，很重要，决定了边缘响应点的分布
    setROI(edge_img, ROI);

#ifdef SHOW_DETAIL
    // 保存ROI边缘图像
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/edgeROI2_img.jpg", edge_img);
#endif

    // 对霍夫直线进行处理
    response_points_.clear(); // 边缘响应点每帧更新一次
    processLanes(hough_lines, edge_img, roi_img, origin_img);

    // 显示检测结果
    // float y_start = float(left_y_min_ > right_y_min_ ? left_y_min_ : right_y_min_);
    float y_start = y_start_ - offset_y_;
    float y_end = roi_img.rows;
    float lx_start = (y_start - laneL.b.get()) / (laneL.k.get() + 1e-6);
    float lx_end = (y_end - laneL.b.get()) / (laneL.k.get() + 1e-6);
    // 绘制左车道直线
    cv::line(image,
             cv::Point2f{lx_start, y_start + offset},
             cv::Point2f{lx_end, y_end + offset},
             cv::Scalar(0, 0, 255), 2);
    std::cout << " left: k = " << laneL.k.get() << " b = " << laneL.b.get() << std::endl;
    float rx_start = (y_start - laneR.b.get()) / (laneR.k.get() + 1e-6);
    float rx_end = (y_end - laneR.b.get()) / (laneR.k.get() + 1e-6);
    // 绘制右车道直线
    cv::line(image,
             cv::Point2f{rx_start, y_start + offset},
             cv::Point2f{rx_end, y_end + offset},
             cv::Scalar(255, 0, 0), 2);
    std::cout << "right: k = " << laneR.k.get() << " b = " << laneR.b.get() << std::endl;
    // 绘制曲线车道
    std::vector<double> left_curve_params, right_curve_params;
    left_curve_params.push_back(laneL.a0.get());
    left_curve_params.push_back(laneL.a1.get());
    left_curve_params.push_back(laneL.a2.get());
    left_curve_params.push_back(laneL.a3.get());
    drawCurve(image, left_curve_params, offset, int(lx_start), int(lx_end), int(y_start), int(y_end), left_curve_points_);
    right_curve_params.push_back(laneR.a0.get());
    right_curve_params.push_back(laneR.a1.get());
    right_curve_params.push_back(laneR.a2.get());
    right_curve_params.push_back(laneR.a3.get());
    drawCurve(image, right_curve_params, offset, int(rx_start), int(rx_end), int(y_start), int(y_end), right_curve_points_);

#ifdef SHOW_DETAIL
    // 绘制曲线车道
    cv::imwrite("/auto-driving-tec/data/lane_detect/result/curve_img.jpg", image);
#endif

    // 绘制可行驶区域
    fillCurvePoly(image, left_curve_points_, right_curve_points_);

    return true;
}

bool LaneDetect::processLanes(std::vector<cv::Vec4i> &lines, cv::Mat &edge_img, cv::Mat &roi_img, cv::Mat &origin_image)
{
    /*
    ** 将经过霍夫变换获得的线分为左右两边
    ** 分别对左右两边的线处理，获得左右车道线
    */

    std::vector<Lane> left, right;

    // 对霍夫变换检测的线进行处理，将左右两边的线进行分类
    int num_lines = static_cast<int>(lines.size());
    for (int i = 0; i < num_lines; ++i)
    {
        // 取出检测到的线段，从起点到终点（x0, y0, x1, y1)
        cv::Vec4i line = lines[i];
        cv::Point2i p0{line[0], line[1]};
        cv::Point2i p1{line[2], line[3]};
        int dx = p1.x - p0.x;                       // x1-x0
        int dy = p1.y - p0.y;                       // y1-y0
        float angle = atan2f(dy, dx) * 180 / CV_PI; // 求线的角度

        // 去除接近于水平的线 (竖直线才可能是车道线)
        if (fabs(angle) <= LINE_REJECT_DEGREES)
        {
            continue;
        }

        // 假设灭点靠近图像水平中心
        // 直线方程: y = kx + b
        dx = (dx == 0) ? 1 : dx;
        float k = dy / (float)dx;
        float b = p0.y - k * p0.x;

        // 根据线的中点位置指定线的边（在左还是在右）
        int mid_x = (p0.x + p1.x) * 0.5;
        int width_center = roi_img.cols * 0.5;
        if (mid_x < width_center)
        {
            left.push_back(Lane(p0, p1, angle, k, b));
        }
        else if (mid_x > width_center)
        {
            right.push_back(Lane(p0, p1, angle, k, b));
        }
    }

    // 将线进行平均
    std::vector<Lane> left_average{left}, right_average{right};
    linesAverage(left_average, origin_image);
    linesAverage(right_average, origin_image);

    left.swap(left_average);
    right.swap(right_average);

    // 显示检测到的线
    int offset = roi_img.rows;
#ifdef SHOW_DETAIL
    {
        cv::Mat tmp_roi_img;
        roi_img.copyTo(tmp_roi_img);
        // 右侧
        std::cout << std::endl
                  << "num lines of right:" << right.size() << std::endl;
        for (int i = 0; i < right.size(); ++i)
        {
            cv::Point2i origin_p0 = right[i].point_start;
            // origin_p0.y += offset;
            cv::Point2i origin_p1 = right[i].point_end;
            // origin_p1.y += offset;
            cv::line(tmp_roi_img, origin_p0, origin_p1, cv::Scalar(180, 0, 0));
        }
        // 左侧
        std::cout << std::endl
                  << "num lines of left:" << left.size() << std::endl;
        for (int i = 0; i < left.size(); ++i)
        {
            cv::Point2i origin_p0 = left[i].point_start;
            // origin_p0.y += offset;
            cv::Point2i origin_p1 = left[i].point_end;
            // origin_p1.y += offset;
            cv::line(tmp_roi_img, origin_p0, origin_p1, cv::Scalar(0, 0, 180));
        }
        // 保存所有直线的图像
        cv::imwrite("/auto-driving-tec/data/lane_detect/result/mulLines_img.jpg", tmp_roi_img);
    }
#endif

    // 处理左边的线
    processSide(left, edge_img, false, origin_image);
    // 处理右边的线
    processSide(right, edge_img, true, origin_image);

    // show results
#ifdef SHOW_DETAIL
    {
        cv::Mat tmp_roi_img;
        roi_img.copyTo(tmp_roi_img);

        int x = tmp_roi_img.cols * 0.55f;
        int x2 = tmp_roi_img.cols;
        cv::line(tmp_roi_img,
                 cv::Point2f{(float)x, float(laneR.k.get() * x + laneR.b.get())},
                 cv::Point2f{(float)x2, float(laneR.k.get() * x2 + laneR.b.get())},
                 cv::Scalar(255, 0, 0), 3);
        x = tmp_roi_img.cols * 0;
        x2 = tmp_roi_img.cols * 0.45f;

        cv::line(tmp_roi_img,
                 cv::Point2f{(float)x, float(laneL.k.get() * x + laneL.b.get())},
                 cv::Point2f{(float)x2, float(laneL.k.get() * x2 + laneL.b.get())},
                 cv::Scalar(0, 0, 255), 3);
        // 保存单线图像
        cv::imwrite("/auto-driving-tec/data/lane_detect/result/oneLine_img.jpg", tmp_roi_img);
    }
#endif
}

bool LaneDetect::processSide(std::vector<Lane> &side_lines, cv::Mat &edge_img, bool is_right, cv::Mat &origin_image)
{
    /*
    **对经过左右边分类的每条直线进行处理，获取位于中心的车道线
    **结果更新在laneR或者laneL中
    */

    Status &side = is_right ? laneR : laneL;
    std::vector<cv::Point2f> &line_points = is_right ? right_points_ : left_points_;
    std::vector<std::vector<cv::Point2f>> &frames_points = is_right ? right_frames_points_ : left_frames_points_;
    std::vector<cv::Point2f> &target_points = is_right ? right_last_points_ : left_last_points_;
    int &y_min = is_right ? right_y_min_ : left_y_min_;

    /* response search */
    int w = edge_img.cols;
    int h = edge_img.rows;
    const int begin_y = 0;
    const int end_y = h - 1;
    const int end_x = is_right ? (w - BORDERX) : BORDERX;
    int mid_x = w / 2;
    int mid_y = h / 2;

    // 建立投票机制，对每条线投票，初始均为0票
    int *votes = new int[side_lines.size()];
    for (std::size_t i = 0; i < side_lines.size(); ++i)
        votes[i] = 0;

    // 遍历每一个图像的y坐标，找到最靠近图像中心的边缘点
    // 哪条线与边缘点匹配，则该直线获得一张投票
    std::vector<cv::Point2i> edge_points;
    for (int y = end_y; y >= begin_y; y -= SCAN_STEP)
    {
        // 寻找响应点，即边缘点，从图像中心开始找
        std::vector<int> rsp;
        findResponses(edge_img, mid_x, end_x, y, rsp);

        if (rsp.size() > 0)
        {
            // 第一个找到的边缘点更靠近图像中心
            int response_x = rsp[0];
            // 用于显示
            response_points_.push_back(cv::KeyPoint{float(response_x), float(y + h), 2});
            // 用于计算
            edge_points.push_back(cv::Point2i{response_x, y});

            float dmin = std::numeric_limits<float>::max();
            float xmin = std::numeric_limits<float>::max();
            int match = -1;

            // 遍历所有线，对每条线进行投票
            int num_lines = static_cast<int>(side_lines.size());
            for (std::size_t i = 0; i < side_lines.size(); ++i)
            {
                // 计算响应点到直线的距离
                float dist = computeDistance(side_lines[i].k, side_lines[i].b, cv::Point2i{response_x, y});
                // 通过直线方程求x坐标
                int x_line = (y - side_lines[i].b) / side_lines[i].k;
                // 图像水平中心与直线上x坐标的距离
                int dist_mid = abs(mid_x - x_line);
                // 过于偏远的点不参与投票
                if (dist > MAX_DISTANCE)
                    continue;
                // 找到距离最靠近图像中心的边缘点的直线
                // 响应点到谁的垂直距离最短，同时距离图像水平中心最近，表明谁最匹配该响应点
                if (match == -1 || (dist <= dmin && dist_mid < xmin))
                {
                    dmin = dist;
                    // 第i条直线是匹配的
                    match = i;
                    xmin = dist_mid;
                    // break;
                }
            }
            // 如果在该直线上找到匹配点，则该直线获得一张投票
            if (match != -1)
            {
                votes[match] += 1;
            }
        }
    }

    int best_match = -1;
    int min_dist = std::numeric_limits<int>::max();
    for (std::size_t i = 0; i < side_lines.size(); ++i)
    {
        // 图像中心y坐标对应直线上的x坐标
        int x_line = (mid_y - side_lines[i].b) / side_lines[i].k;
        // 直线上x坐标与图像中心x坐标的差值
        int dist = abs(mid_x - x_line);
        // 哪一条直线获得投票更多且距离图像水平中心更近，则更新最佳匹配
        if (best_match == -1 || (votes[i] > votes[best_match] /*&& dist < min_dist*/))
        {
            best_match = i;
            min_dist = dist;
        }
    }

#ifdef SHOW_DETAIL
    {
        cv::Mat tmp_img;
        origin_image.copyTo(tmp_img);
        cv::drawKeypoints(tmp_img, response_points_, tmp_img);
        // 保存响应点分布图像
        cv::imwrite("/auto-driving-tec/data/lane_detect/result/response_img.jpg", tmp_img);
    }
#endif

    std::vector<double> curve_params;
    if (best_match != -1)
    {
        // 最佳匹配的直线
        Lane *best = &side_lines[best_match];

        // 判断响应点与最佳直线的距离
        std::vector<cv::Point2f> points;
        for (size_t i = 0; i < edge_points.size(); ++i)
        {
            // 该距离为像素距离
            float dist = computeDistance(best->k, best->b, edge_points[i]);
            // 若响应点到直线的距离小于阈值，则为好点
            if (abs(dist) < MIN_DISTANCE)
            {
                points.push_back(cv::Point2f{float(edge_points[i].x), float(edge_points[i].y)});
            }
        }

        // 在靠近车的区域补充点
        std::vector<cv::Point2f> enhance_points;
        addPoints(edge_img, points, best->k, best->b, enhance_points, y_min);

#ifdef SHOW_DETAIL
        cv::Mat tmp_img;
        origin_image.copyTo(tmp_img);
        drawPoints(tmp_img, points, 0, origin_image.rows / 2, cv::Scalar(0, 255, 0));
        // 保存初始点分布
        cv::imwrite("/auto-driving-tec/data/lane_detect/result/points1_img.jpg", tmp_img);
        drawPoints(tmp_img, enhance_points, 0, origin_image.rows / 2, cv::Scalar(0, 255, 255));
        // 保存补充点分布
        cv::imwrite("/auto-driving-tec/data/lane_detect/result/points2_img.jpg", tmp_img);
#endif

        // 曲线拟合
        fitCurve(origin_image, enhance_points, curve_params);

        // 保存当前帧的响应点(包含补充点)
        line_points.clear();
        line_points.swap(enhance_points);
    }

    if (best_match != -1)
    {
        // 最佳匹配的直线
        Lane *best = &side_lines[best_match];
        // 与历史直线系数的差值
        float k_diff = fabs(best->k - side.k.get());
        float b_diff = fabs(best->b - side.b.get());
        // 判断是否需要更新车道线
        bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side.reset;
        // update_ok = true;
#ifdef SHOW_DETAIL
        printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n",
               (is_right ? "Right" : "Left"), k_diff, b_diff, (update_ok ? "no" : "yes"));
#endif
        if (update_ok)
        {
            /* update is in valid bounds */
            side.k.add(best->k);
            side.b.add(best->b);
            if (curve_params.size() == 4)
            {
                side.a0.add(curve_params[0]);
                side.a1.add(curve_params[1]);
                side.a2.add(curve_params[2]);
                side.a3.add(curve_params[3]);
            }
            side.reset = false;
            side.lost = 0;
        }
        else
        {
            /* can't update, lanes flicker periodically, start counter for partial reset! */
            side.lost++;
            if (side.lost >= MAX_LOST_FRAMES && !side.reset)
            {
                side.reset = true;
            }
        }
    }
    else
    {
#ifdef SHOW_DETAIL
        printf("no lanes detected - lane tracking lost! counter increased\n");
#endif
        printf("no lanes detected - lane tracking lost! counter increased\n");
        side.lost++;
        if (side.lost >= MAX_LOST_FRAMES && !side.reset)
        {
            /* do full reset when lost for more than N frames */
            side.reset = true;
            side.k.clear();
            side.b.clear();
        }
    }

    delete[] votes;
    return true;
}

void LaneDetect::linesAverage(std::vector<Lane> &lines, cv::Mat &origin_image)
{
    std::vector<Lane> lines_new;
    std::vector<int> lines_ids;
    for (std::size_t i = 0; i < lines.size(); ++i)
    {
        lines_ids.push_back(i);
    }
    for (auto ite = lines_ids.begin(); ite != lines_ids.end(); ++ite)
    {
        Lane line{lines[(*ite)].point_start, lines[(*ite)].point_end, lines[(*ite)].angle, lines[(*ite)].k, lines[(*ite)].b};
        int count = 1;
        lines_ids.erase(ite);
        ite--;
        // y轴截距与斜率相近
        for (auto ite2 = lines_ids.begin(); ite2 != lines_ids.end(); ++ite2)
        {
            Lane line2 = lines[(*ite2)];
            if (abs(line.b - line2.b) < B_THRESHOLD && abs(line.k - line2.k) < K_THRESHOLD)
            {
                line.k += line2.k;
                line.b += line2.b;
                count++;
                lines_ids.erase(ite2);
                ite2--;
            }
        }
        line.k /= count;
        line.b /= count;
        int x1 = 0, x2 = origin_image.cols;
        line.point_start = cv::Point2i{x1, int(line.k * x1 + line.b)};
        line.point_end = cv::Point2i{x2, int(line.k * x2 + line.b)};
        line.angle = atanf(line.k) * 180 / CV_PI; // 求线的角度
        lines_new.push_back(line);
    }
    lines.swap(lines_new);
}

bool LaneDetect::findResponses(cv::Mat &edge_img, int begin_x, int end_x, int y, std::vector<int> &rsp)
{
    /*
    ** 根据y坐标与边缘图，找到y坐标对应的所有边缘点(区分左右方向)
    */

    // 步长，区分左右
    int step = (end_x > begin_x) ? 1 : -1;
    // 范围，区分左右
    int range = (end_x > begin_x) ? (end_x - begin_x + 1) : (begin_x - end_x + 1);

    for (int x = begin_x; range > 0; x += step, range--)
    {
        // 寻找白点，黑点就跳过
        if (edge_img.at<uchar>(y, x) <= BW_TRESHOLD)
            continue;
        // 找到的第一个白点为x
        // 判断下一个点是否为白点
        int idx = x + step;
        range--;
        // 确定是否为连续白点
        // 111111000001111
        // 1为白点，0为黑点
        while (range > 0 && edge_img.at<uchar>(y, idx) > BW_TRESHOLD)
        {
            idx += step;
            range--;
        }
        // 连续白点后再次回到黑点
        if (edge_img.at<uchar>(y, idx) <= BW_TRESHOLD)
        {
            // 则x位置为一个响应点，保存
            rsp.push_back(x);
        }

        x = idx;
    }
    // 这里有一个疑问：
    // 如果是车道线是虚线，找到的响应点就是非车道线上的点。
}

void LaneDetect::setROI(cv::Mat &input, std::vector<cv::Point2i> &roi)
{
    cv::Mat mask = cv::Mat::zeros(input.rows, input.cols, CV_8UC1);
    cv::fillPoly(mask, roi, cv::Scalar(255));
    cv::Mat tmp;
    input.copyTo(tmp, mask);
    tmp.copyTo(input);
    // cv::namedWindow("poly");
    // cv::imshow("poly", mask);
    // cv::namedWindow("origin");
    // cv::imshow("origin", input);
    // cv::waitKey();
}

void LaneDetect::seclectWhite(cv::Mat &input, cv::Mat &output, int threshold)
{
    int channels = input.channels();
    int rows = input.rows;
    int cols = input.cols;
    // 转换颜色空间
    cv::Mat hls;
    cv::cvtColor(input, hls, cv::COLOR_BGR2HLS);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8U);
    // 分离L通道
    cv::Mat l_channel = cv::Mat::zeros(input.rows, input.cols, CV_8U);
    cv::Mat dst[]{l_channel};
    int fromTo[] = {1, 0};
    cv::mixChannels(&hls, 1, dst, 1, fromTo, 1);

    double max_value, min_value;
    cv::minMaxLoc(l_channel, &min_value, &max_value);
    l_channel *= (255 / max_value);
    // 提取白色
    int i, j;
    uchar *p;
    for (i = 0; i < rows; ++i)
    {
        p = l_channel.ptr<uchar>(i);
        for (j = 0; j < cols; ++j)
        {
            if (p[j] > threshold)
            {
                output.at<uchar>(i, j) = 255;
            }
        }
    }

    // cv::namedWindow("input");
    // cv::imshow("input", input);
    // cv::namedWindow("L_Channel");
    // cv::imshow("L_Channel", l_channel);
    // cv::namedWindow("white");
    // cv::imshow("white", output);
    // cv::waitKey();
}

void LaneDetect::seclectYellow(cv::Mat &input, cv::Mat &output, int threshold)
{
    int channels = input.channels();
    int rows = input.rows;
    int cols = input.cols;
    // 转换颜色空间
    cv::Mat lab;
    cv::cvtColor(input, lab, cv::COLOR_BGR2Lab);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8U);
    // 分离L通道
    cv::Mat b_channel = cv::Mat::zeros(input.rows, input.cols, CV_8U);
    cv::Mat dst[]{b_channel};
    int fromTo[] = {2, 0};
    cv::mixChannels(&lab, 1, dst, 1, fromTo, 1);
    double max_value, min_value;
    cv::minMaxLoc(b_channel, &min_value, &max_value);
    if (max_value > 100)
        b_channel *= (255 / max_value);
    // 提取黄色
    int i, j;
    uchar *p;
    for (i = 0; i < rows; ++i)
    {
        p = b_channel.ptr<uchar>(i);
        for (j = 0; j < cols; ++j)
        {
            if (p[j] > threshold)
            {
                output.at<uchar>(i, j) = 255;
            }
        }
    }
    // cv::namedWindow("input");
    // cv::imshow("input", input);
    // cv::namedWindow("b_Channel");
    // cv::imshow("b_Channel", b_channel);
    // cv::namedWindow("yellow");
    // cv::imshow("yellow", output);
    // cv::waitKey();
}

float LaneDetect::computeDistance(cv::Point2i line_p0, cv::Point2i line_p1, cv::Point2i point)
{
    // 直线两点之间的距离
    float length_line = sqrt(float(line_p1.x - line_p0.x) * float(line_p1.x - line_p0.x) +
                             float(line_p1.y - line_p0.y) * float(line_p1.y - line_p0.y));
    float inv_length = 1.f / (length_line + 1e-6f);
    // 向量pp0
    cv::Vec2f pp0{float(point.x - line_p0.x), float(point.y - line_p0.y)};
    // 直线p1p0的单位向量
    cv::Vec2f norm_p1p0{float(line_p1.x - line_p0.x) * inv_length, float(line_p1.y - line_p0.y) * inv_length};
    // 向量pp0到直线的投影距离
    float proj_length = norm_p1p0.dot(pp0);

    float dist{};
    if (proj_length >= length_line)
    {
        dist = sqrt(float(line_p1.x - point.x) * float(line_p1.x - point.x) +
                    float(line_p1.y - point.y) * float(line_p1.y - point.y));
    }
    else if (proj_length < 0)
    {
        dist = sqrt(float(line_p0.x - point.x) * float(line_p0.x - point.x) +
                    float(line_p0.y - point.y) * float(line_p0.y - point.y));
    }
    else
    {
        // 投影点
        cv::Point2f p_proj{line_p0.x + norm_p1p0[0] * proj_length, line_p0.y + norm_p1p0[1] * proj_length};

        dist = sqrt(float(p_proj.x - point.x) * float(p_proj.x - point.x) +
                    float(p_proj.y - point.y) * float(p_proj.y - point.y));
    }
    return dist;
}

float LaneDetect::computeDistance(float k, float b, cv::Point2i point)
{
    float dist{};
    float y = float(point.y);
    float x = float(point.x);
    dist = abs(y - (k * x + b)) / sqrt(1.f + k * k);
    return dist;
}

void LaneDetect::slideWindow(cv::Mat &edge_img, cv::Point2i point, float k, float b, std::vector<cv::Point2f> &points_line)
{
    int x_start{point.x}, y_start{point.y};
    int windowSize_x{30}, windowSize_y{10};
    int haldSize_x{windowSize_x / 2};
    int rows = edge_img.rows;
    int cols = edge_img.cols;
    points_line.clear();
    for (int i = 0; i < rows; i += windowSize_y)
    {
        float x_coord = 0, y_coord = 0, count = 0;
        for (int x = x_start - haldSize_x; x < x_start + haldSize_x; ++x)
        {
            for (int y = y_start - windowSize_y + 1; y <= y_start; ++y)
            {
                // 如果存在边缘点
                if (edge_img.at<uchar>(y, x) > 200)
                {
                    x_coord += x;
                    y_coord += y;
                    count++;
                }
            }
        }
        float x_tmp, y_tmp;
        if (count == 0)
        {
            y_tmp = y_start - windowSize_y;
            x_tmp = (y_tmp - b) / (k + 1e-6);

            if (abs(x_tmp - x_start) > 100 || x_tmp < 0 || x_tmp >= cols)
                x_tmp = x_start;
            // points_line.push_back(cv::Point2f{x_tmp, y_tmp});
        }
        else
        {
            x_tmp = x_coord / count;
            y_tmp = y_coord / count;
            points_line.push_back(cv::Point2f{x_tmp, y_tmp});
        }

        x_start = (int)x_tmp;
        y_start -= windowSize_y;
    }
    // cv::Mat tmp_img;
    // edge_img.copyTo(tmp_img);
    // int num_points = static_cast<int>(points_line.size());
    // for (size_t i = 0; i < points_line.size(); ++i)
    // {
    //     cv::circle(tmp_img, points_line[i], 3, cv::Scalar(180), 3);
    // }
    // std::string window_name = "show_line_points";
    // cv::namedWindow(window_name, cv::WINDOW_FREERATIO);
    // cv::imshow(window_name, tmp_img);
    // cv::waitKey();
}

void LaneDetect::addPoints(const cv::Mat &image, const std::vector<cv::Point2f> &target_points, float k, float b,
                           std::vector<cv::Point2f> &points_line, int &y_min)
{
    int num_points = static_cast<int>(target_points.size());
    if (num_points == 0)
        return;
    int y_start{image.rows};
    float y_max = target_points[0].y;
    y_min = y_max;
    points_line.clear();
    for (int i = 0; i < num_points; ++i)
    {
        y_max = y_max > target_points[i].y ? y_max : target_points[i].y;
        y_min = y_min < target_points[i].y ? y_min : target_points[i].y;
        points_line.push_back(target_points[i]);
    }
    for (int i = y_start; i > y_max; i -= SCAN_STEP)
    {
        cv::Point2f add_point{float((float(i) - b) / (k + 1e-6)), (float)i};
        points_line.push_back(add_point);
    }
}

void LaneDetect::fitCurve(std::vector<double> &x, std::vector<double> &y, std::vector<double> &curve_params)
{
    // columns is 3, rows is x.size()
    int num_points = static_cast<int>(x.size());
    if (x.size() != y.size() || num_points == 0)
        return;
    cv::Mat A = cv::Mat::zeros(cv::Size(3, x.size()), CV_64FC1);
    for (int i = 0; i < x.size(); i++)
    {
        A.at<double>(i, 0) = 1;
        A.at<double>(i, 1) = x[i];
        A.at<double>(i, 2) = x[i] * x[i];
    }

    cv::Mat b = cv::Mat::zeros(cv::Size(1, y.size()), CV_64FC1);
    for (int i = 0; i < y.size(); i++)
    {
        b.at<double>(i, 0) = y[i];
    }

    cv::Mat c;
    c = A.t() * A;
    cv::Mat d;
    d = A.t() * b;

    cv::Mat result = cv::Mat::zeros(cv::Size(1, 3), CV_64FC1);
    cv::solve(c, d, result);
    // std::cout << "A = " << A << std::endl;
    // std::cout << "b = " << b << std::endl;
    // std::cout << "result = " << result << std::endl;
    double a0 = result.at<double>(0, 0);
    double a1 = result.at<double>(1, 0);
    double a2 = result.at<double>(2, 0);
    curve_params.clear();
    curve_params.push_back(a0);
    curve_params.push_back(a1);
    curve_params.push_back(a2);
    // std::cout << "对称轴：" << -a1 / a2 / 2 << std::endl;
    std::cout << "二次曲线拟合方程：y = " << a0 << " + (" << a1 << "x) + (" << a2 << "x^2)" << std::endl;
}

void LaneDetect::fitCurve(cv::Mat &image, const std::vector<cv::Point2f> &points, std::vector<double> &curve_params)
{
    // 利用RANSAC算法进行三次曲线拟合
    // y=a0+a1*x+a2*x^2+a3*x^3
    int num_points = static_cast<int>(points.size());
    if (num_points <= 3)
        return;
    int num_groups = CURVE_NUM_GROUPS;
    std::vector<std::vector<int>> ids;
    srand((int)time(0));
    int num_control_points = CURVE_NUM_POINTS;
    if (num_control_points >= num_points)
    {
        num_control_points = num_points;
        num_groups = 1;
    }

    // 随机选择n个点,生成多种组合
    for (int i = 0; i < num_groups; ++i)
    {
        std::vector<int> id_group;
        for (int j = 0; j < num_control_points; ++j)
        {
            int id = rand() % num_points;
            id_group.push_back(id);
        }
        ids.push_back(id_group);
    }
    // 对每一种组合进行曲线拟合
    double dist_min = std::numeric_limits<double>::max();
    std::vector<std::vector<double>> params;
    int best_group = -1;
    for (int i_group = 0; i_group < num_groups; ++i_group)
    {
        // 求解三次曲线的系数
        std::vector<int> &id_group = ids[i_group];
        cv::Mat X = cv::Mat::zeros(num_control_points, 4, CV_64FC1);
        cv::Mat Y = cv::Mat::zeros(num_control_points, 1, CV_64FC1);
        std::vector<cv::Point2f> points_use;
        for (int i = 0; i < num_control_points; i++)
        {
            double x_coord = double(points[id_group[i]].x);
            double y_coord = double(points[id_group[i]].y);
            points_use.push_back(points[id_group[i]]);
            X.at<double>(i, 0) = 1;
            X.at<double>(i, 1) = x_coord;
            X.at<double>(i, 2) = x_coord * x_coord;
            X.at<double>(i, 3) = x_coord * x_coord * x_coord;
            Y.at<double>(i, 0) = y_coord;
        }
        // 伪逆法求方程系数
        cv::Mat XTX = X.t() * X;
        cv::Mat XTY = X.t() * Y;
        cv::Mat res = m_Bezier3_inv_ * XTX.inv() * XTY;
        double a0 = res.at<double>(0);
        double a1 = res.at<double>(1);
        double a2 = res.at<double>(2);
        double a3 = res.at<double>(3);
        std::vector<double> cur_params;
        cur_params.push_back(a0);
        cur_params.push_back(a1);
        cur_params.push_back(a2);
        cur_params.push_back(a3);
        params.push_back(cur_params);

        // 数据点结果
        // std::vector<cv::Point2f> res_points;
        // for (int i = 0; i < num_control_points; i++)
        // {
        //     double x_coord = double(points[id_group[i]].x);
        //     double y_coord = getBezierValue(x_coord, cur_params);
        //     res_points.push_back(cv::Point2f{float(x_coord), float(y_coord)});
        //     std::cout << "(" << x_coord << "," << y_coord << ")";
        // }
        // std::cout << std::endl;

        // int offset = image.rows / 2;
        // drawCurve(image, cur_params, offset, offset + 100);
        // drawCurve(image, points_use, offset);
        // //drawCurve(image, res_points, offset);
        // std::string window_name = "test_curve";
        // cv::namedWindow(window_name);
        // cv::imshow(window_name, image);
        // cv::waitKey();

        // 统计所有点到曲线的距离
        double dist = 0;
        for (int i_point = 0; i_point < num_points; ++i_point)
        {
            double x = double(points[i_point].x);
            double y = double(points[i_point].y);
            double y_curve = getBezierValue(x, cur_params);
            dist += abs(abs(y) - abs(y_curve));
        }

        if (dist_min >= dist)
        {
            dist_min = dist;
            best_group = i_group;
        }
    }

    if (best_group != -1)
    {
        curve_params.clear();
        for (int i = 0; i < 4; ++i)
        {
            curve_params.push_back(params[best_group][i]);
        }
    }
}

void LaneDetect::drawCurve(cv::Mat &image, const std::vector<double> &curve_params, int offset_y, int x_start, int x_end, int y_start, int y_end, std::vector<cv::Point2f> &curve_points)
{
    curve_points.clear();
    int rows = image.rows;
    int cols = image.cols;
    int x1, x2;
    if (x_start > x_end)
    {
        x1 = x_end;
        x2 = x_start;
    }
    else
    {
        x1 = x_start;
        x2 = x_end;
    }

    for (int x = x1; x < x2; x += 5)
    {
        double y = getBezierValue(x, curve_params);
        if (y < y_start || y > y_end)
            continue;
        y += offset_y;
        cv::circle(image, cv::Point2f{(float)x, (float)y}, 2, cv::Scalar(0, 255, 0), 2);
        curve_points.push_back(cv::Point2f{(float)x, (float)y});
    }

    // std::string window_name = "curve";
    // cv::namedWindow(window_name);
    // cv::imshow(window_name, image);
    // cv::waitKey();
}

void LaneDetect::drawCurve(cv::Mat &image, const std::vector<cv::Point2f> &line_points, int offset_y)
{
    int num_points = static_cast<int>(line_points.size());
    for (int i = 0; i < num_points; ++i)
    {
        cv::circle(image, cv::Point2f{line_points[i].x, line_points[i].y + offset_y}, 2, cv::Scalar(0, 255, 255), 2);
    }
}

void LaneDetect::fillCurvePoly(cv::Mat &image, const std::vector<cv::Point2f> &left_curve_points, const std::vector<cv::Point2f> &right_curve_points)
{
    int width = image.cols;
    int height = image.rows;
    int num_left_points = static_cast<int>(left_curve_points.size());
    int num_right_points = static_cast<int>(right_curve_points.size());
    if (num_left_points == 0 || num_right_points == 0)
        return;
    std::vector<cv::Point> poly_points;
    for (int i = 0; i < num_left_points; ++i)
    {
        cv::Point2f p = left_curve_points[i];
        if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
            poly_points.push_back(cv::Point{(int)p.x, (int)p.y});
    }
    for (int i = 0; i < num_right_points; ++i)
    {
        cv::Point2f p = right_curve_points[i];
        if (p.x >= 0 && p.x < width && p.y >= 0 && p.y < height)
            poly_points.push_back(cv::Point{(int)p.x, (int)p.y});
    }
    cv::Mat poly_mask = cv::Mat::zeros(image.size(), image.type());
    cv::fillPoly(poly_mask, poly_points, cv::Scalar(0, 100, 0));
    image += 0.7 * poly_mask;
}

void LaneDetect::drawPoints(cv::Mat &image, const std::vector<cv::Point2f> &points, int offset_x, int offset_y, cv::Scalar color)
{
    int num_points = static_cast<int>(points.size());
    for (int i = 0; i < num_points; ++i)
    {
        cv::circle(image, cv::Point2f{points[i].x + offset_x, points[i].y + offset_y}, 2, color, 2);
    }
}

double LaneDetect::getBezierValue(double x, const std::vector<double> &curve_params)
{
    double a0 = curve_params[0];
    double a1 = curve_params[1];
    double a2 = curve_params[2];
    double a3 = curve_params[3];
    cv::Mat P = cv::Mat::zeros(4, 1, CV_64F);
    P = (cv::Mat_<double>(4, 1) << a0, a1, a2, a3);
    cv::Mat X = cv::Mat::zeros(1, 4, CV_64F);
    X = (cv::Mat_<double>(1, 4) << 1, x, x * x, x * x * x);
    cv::Mat Y = X * m_Bezier3_ * P;
    double y = Y.at<double>(0);
    return y;
}

void LaneDetect::surfMatch(cv::Mat &cur_img, cv::Mat &last_img, cv::Mat &H_matrix)
{
    cv::Ptr<cv::xfeatures2d::SURF> surf; // 创建方式和OpenCV2中的不一样,并且要加上命名空间xfreatures2d
                                         // 否则即使配置好了还是显示SURF为未声明的标识符
    surf = cv::xfeatures2d::SURF::create(800);

    cv::BFMatcher matcher; // 实例化一个暴力匹配器
    cv::Mat feature_cur_img, feature_last_img;
    // 特征点
    std::vector<cv::KeyPoint> key1, key2;
    // DMatch是用来描述匹配好的一对特征点的类，包含这两个点之间的相关信息
    // 比如左图有个特征m，它和右图的特征点n最匹配，这个DMatch就记录它俩最匹配，并且还记录m和n的
    // 特征向量的距离和其他信息，这个距离在后面用来做筛选
    std::vector<cv::DMatch> matches;
    // 输入图像，输入掩码用于屏蔽源图像中的特定区域，输入特征点矢量数组 ，存放所有特征点的描述向量
    // feature_cur_img行数为特征点的个数，列数为每个特征向量的尺寸，SURF是64（维）
    surf->detectAndCompute(cur_img, cv::Mat(), key1, feature_cur_img);
    surf->detectAndCompute(last_img, cv::Mat(), key2, feature_last_img);

    matcher.match(feature_cur_img, feature_last_img, matches); // 匹配，数据来源是特征向量，结果存放在DMatch类型里面

    // sort函数对数据进行升序排列
    sort(matches.begin(), matches.end());
    std::vector<cv::DMatch> good_matches;
    int ptsPairs = std::min(50, (int)(matches.size() * 0.15));

    std::cout << "point pairs: " << ptsPairs << std::endl;

    // 筛选匹配点，根据match里面特征对的距离从小到大排序
    // 保留好的特征点，剔除误匹配点
    for (int i = 0; i < ptsPairs; i++)
    {
        good_matches.push_back(matches[i]);
    }
    // 计算图像配准点
    std::vector<cv::Point2f> image_Points1, image_Points2;

    for (int i = 0; i < good_matches.size(); i++)
    {
        image_Points1.push_back(key1[good_matches[i].queryIdx].pt);
        image_Points2.push_back(key2[good_matches[i].trainIdx].pt);
    }

    // drawMatches这个函数直接画出摆在一起的图
    // 绘制匹配点
    // cv::Mat out_img;
    // cv::drawMatches(cur_img, key1, last_img, key2, good_matches, out_img,
    //                 cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
    //                 cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // std::string window_name = "SURF Match";
    // cv::namedWindow(window_name);
    // cv::imshow(window_name, out_img);
    // cv::waitKey();

    // 获取图像1到图像2的投影映射矩阵 尺寸为3*3，剔除误配点
    H_matrix = cv::findHomography(image_Points2, image_Points1, cv::RANSAC);
    // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差
    // Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);

    // 输出映射矩阵
    std::cout << "变换矩阵为：\n"
              << H_matrix << std::endl;

    // 图像配准
    // FourCorners corners;
    // CalcCorners(H_matrix, last_img, corners);
    // cv::Mat imageTransform1, imageTransform2;
    // cv::warpPerspective(last_img, imageTransform2, H_matrix, cv::Size(cur_img.cols, std::max(corners.left_bottom.y, corners.right_bottom.y)));
    // // warpPerspective(a, imageTransform2, adjustMat*homo, Size(b.cols*1.3, b.rows*1.8));

    // // 创建拼接后的图,需提前计算图的大小
    // // 取最右点的长度为拼接图的长度
    // int dst_width = cur_img.cols;
    // int dst_height = imageTransform2.rows;
    // cv::Mat dst(dst_height, dst_width, CV_8UC3);
    // dst.setTo(0);
    // imageTransform2.copyTo(dst(cv::Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
    // cur_img.copyTo(dst(cv::Rect(0, 0, cur_img.cols, cur_img.rows)));
    // cv::namedWindow("current");
    // cv::imshow("current", cur_img);
    // cv::namedWindow("last");
    // cv::imshow("last", last_img);
    // cv::namedWindow("transform");
    // cv::imshow("transform", imageTransform2);
    // cv::namedWindow("surf_result");
    // cv::imshow("surf_result", dst);
    // cv::waitKey();
}

void LaneDetect::CalcCorners(const cv::Mat &H, const cv::Mat &src, FourCorners &corners)
{
    double v2[] = {0, 0, 1};                  // 左上角
    double v1[3];                             // 变换后的坐标值
    cv::Mat V2 = cv::Mat(3, 1, CV_64FC1, v2); // 列向量，CV_64FC164位浮点数，通道为1
    cv::Mat V1 = cv::Mat(3, 1, CV_64FC1, v1); // 列向量
    V1 = H * V2;

    // 左上角(0,0,1)
    corners.left_top.x = v1[0] / v1[2];
    corners.left_top.y = v1[1] / v1[2];

    // 左下角(0,src.rows,1)
    v2[0] = 0;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2); // 列向量
    V1 = cv::Mat(3, 1, CV_64FC1, v1); // 列向量
    V1 = H * V2;
    corners.left_bottom.x = v1[0] / v1[2];
    corners.left_bottom.y = v1[1] / v1[2];

    // 右上角(src.cols,0,1)
    v2[0] = src.cols;
    v2[1] = 0;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2); // 列向量
    V1 = cv::Mat(3, 1, CV_64FC1, v1); // 列向量
    V1 = H * V2;
    corners.right_top.x = v1[0] / v1[2];
    corners.right_top.y = v1[1] / v1[2];

    // 右下角(src.cols,src.rows,1)
    v2[0] = src.cols;
    v2[1] = src.rows;
    v2[2] = 1;
    V2 = cv::Mat(3, 1, CV_64FC1, v2); // 列向量
    V1 = cv::Mat(3, 1, CV_64FC1, v1); // 列向量
    V1 = H * V2;
    corners.right_bottom.x = v1[0] / v1[2];
    corners.right_bottom.y = v1[1] / v1[2];
}

void LaneDetect::pointsTransform(std::vector<std::vector<cv::Point2f>> &frames_points, const cv::Mat &H_matrix, int offset_y,
                                 std::vector<cv::Point2f> &target_points)
{
    target_points.clear();
    if (frames_points.size() == 0)
        return;
    if (H_matrix.rows != 3)
        return;

    for (size_t i = 0; i < frames_points.size(); ++i)
    {
        for (size_t j = 0; j < frames_points[i].size(); ++j)
        {
            cv::Mat coord = cv::Mat::zeros(3, 1, H_matrix.type());
            coord.at<double>(0, 0) = double(frames_points[i][j].x);
            coord.at<double>(1, 0) = double(frames_points[i][j].y + offset_y);
            coord.at<double>(2, 0) = 1.0;
            cv::Mat res = H_matrix * coord;
            float new_x = (float)res.at<double>(0) / (float)res.at<double>(2);
            float new_y = (float)res.at<double>(1) / (float)res.at<double>(2) - offset_y;
            target_points.push_back(cv::Point2f{new_x, new_y});
            frames_points[i][j].x = new_x;
            frames_points[i][j].y = new_y;
        }
    }
}