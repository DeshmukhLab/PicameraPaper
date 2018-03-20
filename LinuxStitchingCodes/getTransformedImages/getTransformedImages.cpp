/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//
//M*/


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
#include <typeinfo>

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

// Default command line args
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 0.85f;
string features_type = "orb";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
string warp_type = "plane";
float match_conf = 0.3f;
string result_name = "result.jpg";
int range_width = -1;
string stitchingParamFileName = "stitchingParam.yml";

/**function to get warped and registered image
Input: image, mask, corner point, roi, roi mask, output image reference, output image mask reference
**/
void getWarpedRegisteredImage(InputArray _img, InputArray _mask, Point tl, Rect dst_roi_, Mat dst_mask_, Mat dst_)
{
    int corner_x, corner_y;
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    //update the corner points for each image
    corner_x = tl.x - dst_roi_.x;
    corner_y = tl.y - dst_roi_.y;

    //update output image and image mask with corners updates
    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(corner_y + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(corner_y + y);

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[corner_x + x] = src_row[x];
            dst_mask_row[corner_x + x] |= mask_row[x];
        }
    }
}

/* function to get the transformed Images from each camera using the pre calculated registration data
INPUT: vector of input images 
OUTPUT: vector of transformed+warped images (size same as input image size)
*/
vector<Mat> getTransformedImages(vector<Mat> InputImage, int num_images)
{
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    Mat full_img, img;
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;
    vector<CameraParams> cameras(num_images);

    //update the each input image size depending on the 'work_megapix' variable
    for (int i = 0; i < num_images; ++i)
    {
        full_img = InputImage.at(i);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            LOGLN("Can't open image ");
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }

        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }
    full_img.release();
    img.release();

    //load the camera param calculated using 'getStitchingParam'
    for (int i = 0; i < num_images; ++i)
    {
        Mat K, R, t;
        double ppx, ppy, focal, aspect;
        stringstream camId;
        camId << i+1;
        string fileName = "cam" + camId.str() + ".yml";
        FileStorage fs(fileName, FileStorage::READ);
        fs["K"] >> K;
        fs["R"] >> R;
        fs["t"] >> t;
        fs["ppx"] >> ppx;
        fs["ppy"] >> ppy;
        fs["focal"] >> focal;
        fs["aspect"] >> aspect;
        cameras[i].K() = (Mat)K;
        cameras[i].R = R;
        cameras[i].t = t;
        cameras[i].ppx = (double)ppx;
        cameras[i].ppy = (double)ppy;
        cameras[i].focal = (double)focal;
        cameras[i].aspect = (double)aspect;
        fs.release();
    }

    // Find median focal length
    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    //perform wave correcting or straightening for the rotation parameters for the individual camera
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("Warping images (auxiliary)... ");

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks
    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
    }

    //create warper object which can be plane/affine/cylindrical
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));
    //warp and transform individual camera image
    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Finished warping images");

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    //variable to store the output images and their respective masks
    vector<Mat> transformedImages, transformedImagesMask;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        // Read image and resize it if necessary
        full_img = InputImage.at(img_idx);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        Rect dst_roi, dst_roi_;

        //create blank image and image mask which will store the warped image
        dst_roi = resultRoi(corners, sizes);
        Mat dst_, dst_mask_;
        dst_.create(dst_roi.size(), CV_16SC3);
        dst_.setTo(Scalar::all(0));
        dst_mask_.create(dst_roi.size(), CV_8U);
        dst_mask_.setTo(Scalar::all(0));
        dst_roi_ = dst_roi;

        //conversion to prevent memory overflow (happens sometimes)
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        LOGLN("Transforming Images");
        //save warped and transformed image for each camera and also update the new corner
        getWarpedRegisteredImage(img_warped_s, mask_warped, corners[img_idx], dst_roi_, dst_mask_, dst_);
        LOGLN("Done loading registered images");

        transformedImages.push_back(dst_);
        transformedImagesMask.push_back(dst_mask_);
    }
    //return the list of transformed images
    return transformedImages;
}

int main(int argc, char* argv[])
{
    vector<Mat> inputImages, outputImages;
    vector<Rect> roi;
/**
    Rect cam1Roi = Rect(0,0,625,440);
    Rect cam2Roi = Rect(0,0,640,440);
    Rect cam3Roi = Rect(0,0,640,445);
    Rect cam4Roi = Rect(130,0,510,455);
    Rect cam5Roi = Rect(0,15,600,465);
    Rect cam6Roi = Rect(0,0,640,480);
    Rect cam7Roi = Rect(0,0,640,480);
    Rect cam8Roi = Rect(60,15,580,465);
    roi.push_back(cam1Roi);
    roi.push_back(cam2Roi);
    roi.push_back(cam3Roi);
    roi.push_back(cam4Roi);
    roi.push_back(cam5Roi);
    roi.push_back(cam6Roi);
    roi.push_back(cam7Roi);
    roi.push_back(cam8Roi);
**/
    String imageName;
    for (int i = 1; i < argc; ++i)
    {
        imageName = String(argv[i]);
        Mat img = imread(imageName);
        //img = img(roi.at(i-1));
        inputImages.push_back(img);
    }
    // Check if have enough images
    int num_images = static_cast<int>(inputImages.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }
    //function to get the transformed image for each camera from the input and registration data
    outputImages = getTransformedImages(inputImages, num_images);

    //save the transformed image of each camera 
    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        stringstream camId;
        camId << img_idx + 1;
        String warpedRegisteredImageFileName, warpedRegisteredImageMaskFileName;
        Mat stitchedImage = outputImages[img_idx];
        warpedRegisteredImageFileName = "cam" + camId.str() + "_warped.jpg";
        imwrite(warpedRegisteredImageFileName, stitchedImage);
    }

    return 0;
}
