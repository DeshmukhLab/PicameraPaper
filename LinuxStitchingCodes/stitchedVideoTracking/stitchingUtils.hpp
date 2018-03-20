#ifndef STITCHINGUTILS_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define STITCHINGUTILS_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <numeric>

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace cv;
using namespace cv::detail;
using namespace std;

void registerImages(InputArray _img, InputArray _mask, Point tl, Rect dst_roi_, Mat dst_mask_, Mat dst_);

vector<Mat> getWarpedRegisteredImages(vector<Mat> InputImage);

vector<Point2f> getLedCoordinates(Mat frame);

void rotate_90n(cv::Mat const &src, cv::Mat &dst, int angle);

vector<Point2f> getAveragePosition8cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos, vector<Point2f> &cam4LedPos, vector<Point2f> &cam5LedPos, vector<Point2f> &cam6LedPos, vector<Point2f> &cam7LedPos, vector<Point2f> &cam8LedPos);

vector<Point2f> getAveragePosition4cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos, vector<Point2f> &cam4LedPos);

vector<Point2f> getAveragePosition3cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos);

vector<Point2f> getAveragePosition2cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos);

void writeLedPosToFile(vector<float> red_x, vector<float> red_y, vector<float> green_x, vector<float> green_y, int width, int height, const char* filename);

void writeLedPosToFile4cams(vector<float> cam1_red_x, vector<float> cam1_red_y, vector<float> cam1_green_x, vector<float> cam1_green_y,
                            vector<float> cam2_red_x, vector<float> cam2_red_y, vector<float> cam2_green_x, vector<float> cam2_green_y,
                            vector<float> cam3_red_x, vector<float> cam3_red_y, vector<float> cam3_green_x, vector<float> cam3_green_y,
                            vector<float> cam4_red_x, vector<float> cam4_red_y, vector<float> cam4_green_x, vector<float> cam4_green_y,
                            int width, int height, const char* filename);

void writeLedPosToFile8cams(vector<float> cam1_red_x, vector<float> cam1_red_y, vector<float> cam1_green_x, vector<float> cam1_green_y,
                            vector<float> cam2_red_x, vector<float> cam2_red_y, vector<float> cam2_green_x, vector<float> cam2_green_y,
                            vector<float> cam3_red_x, vector<float> cam3_red_y, vector<float> cam3_green_x, vector<float> cam3_green_y,
                            vector<float> cam4_red_x, vector<float> cam4_red_y, vector<float> cam4_green_x, vector<float> cam4_green_y,
                            vector<float> cam5_red_x, vector<float> cam5_red_y, vector<float> cam5_green_x, vector<float> cam5_green_y,
                            vector<float> cam6_red_x, vector<float> cam6_red_y, vector<float> cam6_green_x, vector<float> cam6_green_y,
                            vector<float> cam7_red_x, vector<float> cam7_red_y, vector<float> cam7_green_x, vector<float> cam7_green_y,
                            vector<float> cam8_red_x, vector<float> cam8_red_y, vector<float> cam8_green_x, vector<float> cam8_green_y,
                            int width, int height, const char* filename);
#endif
