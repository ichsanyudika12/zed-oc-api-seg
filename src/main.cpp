#include <iostream>
#include <string>
#include <iomanip>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

#include "videocapture.hpp"
#include "calibration.hpp"
#include "stereo.hpp"

// HSV Threshold variables
int h_min = 0, h_max = 39;
int s_min = 129, s_max = 255;
int v_min = 59, v_max = 255;

void on_trackbar(int, void*) {}

int main(int argc, char *argv[]) {
    (void)argc;
    (void)argv;

    cv::ocl::setUseOpenCL(false);

    sl_oc::VERBOSITY verbose = sl_oc::VERBOSITY::INFO;

    sl_oc::video::VideoParams params;
    params.res = sl_oc::video::RESOLUTION::VGA;
    params.fps = sl_oc::video::FPS::FPS_30;
    params.verbose = verbose;

    sl_oc::video::VideoCapture cap(params);
    if (!cap.initializeVideo(-1)) {
        std::cerr << "Cannot open camera video capture" << std::endl;
        return EXIT_FAILURE;
    }

    int sn = cap.getSerialNumber();
    std::cout << "Connected to camera sn: " << sn << std::endl;

    std::string calibration_file;
    if (!sl_oc::tools::downloadCalibrationFile(sn, calibration_file)) {
        std::cerr << "Failed to download calibration file." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Calibration file loaded: " << calibration_file << std::endl;

    int w, h;
    cap.getFrameSize(w, h);

    // Calibration and remap maps
    cv::Mat map_left_x, map_left_y, map_right_x, map_right_y;
    cv::Mat cameraMatrix_left, cameraMatrix_right;
    double baseline = 0;
    sl_oc::tools::initCalibration(calibration_file, cv::Size(w / 2, h),
                                  map_left_x, map_left_y,
                                  map_right_x, map_right_y,
                                  cameraMatrix_left, cameraMatrix_right,
                                  &baseline);

    double fx = cameraMatrix_left.at<double>(0, 0);
    double fy = cameraMatrix_left.at<double>(1, 1);
    double cx = cameraMatrix_left.at<double>(0, 2);
    double cy = cameraMatrix_left.at<double>(1, 2);

    std::cout << "Camera Matrix L: \n" << cameraMatrix_left << std::endl;
    std::cout << "Baseline: " << baseline << " mm" << std::endl;

    cv::Mat frameYUV, frameBGR, left_raw, right_raw, left_rect, right_rect;
    cv::Mat left_disp, left_disp_float;

    // Stereo matcher setup
    sl_oc::tools::StereoSgbmPar stereoPar;
    if (!stereoPar.load()) stereoPar.save();

    cv::Ptr<cv::StereoSGBM> left_matcher = cv::StereoSGBM::create(
        stereoPar.minDisparity,
        stereoPar.numDisparities,
        stereoPar.blockSize);

    left_matcher->setMinDisparity(stereoPar.minDisparity);
    left_matcher->setNumDisparities(stereoPar.numDisparities);
    left_matcher->setBlockSize(stereoPar.blockSize);
    left_matcher->setP1(stereoPar.P1);
    left_matcher->setP2(stereoPar.P2);
    left_matcher->setDisp12MaxDiff(stereoPar.disp12MaxDiff);
    left_matcher->setMode(stereoPar.mode);
    left_matcher->setPreFilterCap(stereoPar.preFilterCap);
    left_matcher->setUniquenessRatio(stereoPar.uniquenessRatio);
    left_matcher->setSpeckleWindowSize(stereoPar.speckleWindowSize);
    left_matcher->setSpeckleRange(stereoPar.speckleRange);

    std::cout << "Starting real-time depth measurement...\nPress 'q' to exit" << std::endl;

    // HSV Trackbars
    cv::namedWindow("HSV Trackbars", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("H Min", "HSV Trackbars", &h_min, 179, on_trackbar);
    cv::createTrackbar("H Max", "HSV Trackbars", &h_max, 179, on_trackbar);
    cv::createTrackbar("S Min", "HSV Trackbars", &s_min, 255, on_trackbar);
    cv::createTrackbar("S Max", "HSV Trackbars", &s_max, 255, on_trackbar);
    cv::createTrackbar("V Min", "HSV Trackbars", &v_min, 255, on_trackbar);
    cv::createTrackbar("V Max", "HSV Trackbars", &v_max, 255, on_trackbar);

    // FPS Tracker
    uint64_t last_ts = 0;
    int frame_count = 0;
    float fps = 0;
    auto t_start = std::chrono::high_resolution_clock::now();

    while (true) {
        const sl_oc::video::Frame frame = cap.getLastFrame();
        if (!frame.data || frame.timestamp == last_ts) continue;
        last_ts = frame.timestamp;
        frame_count++;

        // Hitung FPS
        auto t_now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_now - t_start).count();
        if (elapsed > 1.0) {
            fps = frame_count / elapsed;
            frame_count = 0;
            t_start = t_now;
        }

        frameYUV = cv::Mat(frame.height, frame.width, CV_8UC2, frame.data).clone();
        cv::cvtColor(frameYUV, frameBGR, cv::COLOR_YUV2BGR_YUYV);

        right_raw = frameBGR(cv::Rect(frameBGR.cols / 2, 0, frameBGR.cols / 2, frameBGR.rows)).clone();
        left_raw = frameBGR(cv::Rect(0, 0, frameBGR.cols / 2, frameBGR.rows)).clone();

        cv::remap(left_raw, left_rect, map_left_x, map_left_y, cv::INTER_LINEAR);
        cv::remap(right_raw, right_rect, map_right_x, map_right_y, cv::INTER_LINEAR);

        left_matcher->compute(left_rect, right_rect, left_disp);
        left_disp.convertTo(left_disp_float, CV_32F, 1.0 / 16.0);

        cv::Mat depth_map = cv::Mat(left_disp_float.size(), CV_32F, cv::Scalar(0));
        double fx_baseline = fx * baseline;

        cv::Mat hsv, mask;
        cv::cvtColor(left_rect, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(h_min, s_min, v_min), cv::Scalar(h_max, s_max, v_max), mask);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::medianBlur(depth_map, depth_map, 5);

        int center_x = depth_map.cols / 2;
        int center_y = depth_map.rows / 2;
        cv::Rect roi(center_x - 3, center_y - 3, 7, 7);
        roi &= cv::Rect(0, 0, depth_map.cols, depth_map.rows);

        cv::Mat center_region = depth_map(roi);
        cv::Scalar avg_depth = cv::mean(center_region);
        float center_depth = static_cast<float>(avg_depth[0]);
        if (center_depth <= 0 || center_depth > 10000) center_depth = 0;

        double max_area = 0;
        std::vector<cv::Point> max_contour;
        for (const auto& c : contours) {
            double area = cv::contourArea(c);
            if (area > max_area) {
                max_area = area;
                max_contour = c;
            }
        }

        float depth_val = -1;
        if (!max_contour.empty() && max_area > 30) {
            cv::Moments M = cv::moments(max_contour);
            if (M.m00 != 0) {
                cv::Point center(M.m10 / M.m00, M.m01 / M.m00);
                cv::circle(left_rect, center, 5, cv::Scalar(0, 255, 255), -1);

                float disparity = left_disp_float.at<float>(center);
                if (disparity > 1.0 && disparity < stereoPar.numDisparities) {
                    depth_val = (fx * baseline) / disparity;

                    std::cout << "Depth: " << std::fixed << std::setprecision(2) << depth_val / 1000.0f << " m" << std::endl;
                }
            }
        }

        // std::stringstream fps_ss;
        // fps_ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
        // // cv::putText(left_rect, fps_ss.str(), cv::Point(20, 30),
        // //             cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Pov Zed", left_rect);
        cv::imshow("Pov Mask", mask);

        if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "Exiting program..." << std::endl;
    return 0;
}