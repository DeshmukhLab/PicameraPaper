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
#include <numeric>
#include "stitchingUtils.hpp"

#define ENABLE_LOG 1
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

int main(int argc, char* argv[])
{
    //open output csv file
    const char* outputFileName = argv[1];
    int width, height;

    //initlialize variable to hold average red and green led position
    Point2f averageRedLed, averageGreenLed;
    vector<float> red_x, red_y, green_x, green_y;
    vector<float> cam1_red_x, cam1_red_y, cam1_green_x, cam1_green_y;
    vector<float> cam2_red_x, cam2_red_y, cam2_green_x, cam2_green_y;
    vector<float> cam3_red_x, cam3_red_y, cam3_green_x, cam3_green_y;
    vector<float> cam4_red_x, cam4_red_y, cam4_green_x, cam4_green_y;
    vector<float> cam5_red_x, cam5_red_y, cam5_green_x, cam5_green_y;
    vector<float> cam6_red_x, cam6_red_y, cam6_green_x, cam6_green_y;
    vector<float> cam7_red_x, cam7_red_y, cam7_green_x, cam7_green_y;
    vector<float> cam8_red_x, cam8_red_y, cam8_green_x, cam8_green_y;
    vector<Point2f> averageLedPosition;

    //camera capture devices
    vector<VideoCapture> cameraCaptures;
    String videoFileName;
    LOGLN("Loading Video Files from arguments passed");
    for (int i = 2; i < argc; ++i)
    {
        videoFileName = String(argv[i]);
        VideoCapture cap(videoFileName); // open the video file
        if(!cap.isOpened()){ // check if we succeeded
            LOGLN("Cannot open the video file");
            return -1;
        }
        //save the loaded capture devices
        cameraCaptures.push_back(cap);
    }
    LOGLN("Started analyzing the position data");
    while(true)
    {
        //Loading each video file frame
        Mat cam1Frame, cam2Frame, cam3Frame, cam4Frame, cam5Frame, cam6Frame, cam7Frame, cam8Frame;
        vector<Mat> inputFrames, outputFrames;
        //get new frame from each capture devices
        cameraCaptures.at(0).read(cam1Frame);
        cameraCaptures.at(1).read(cam2Frame);
        cameraCaptures.at(2).read(cam3Frame);
        cameraCaptures.at(3).read(cam4Frame);
        cameraCaptures.at(4).read(cam5Frame);
        cameraCaptures.at(5).read(cam6Frame);
        cameraCaptures.at(6).read(cam7Frame);
        cameraCaptures.at(7).read(cam8Frame);

        //check if any camera frame is empty, if empty stop processing
	if( (cam1Frame.empty()) || (cam2Frame.empty()) || (cam3Frame.empty()) || (cam4Frame.empty()) || (cam5Frame.empty()) || (cam6Frame.empty()) || (cam7Frame.empty()) || (cam8Frame.empty()) ){
            LOGLN("End of File Reached or Camera Frame is empty");
            break;
        }

        //if frames are not empty save them to input Frames vector
        inputFrames.push_back(cam1Frame);
        inputFrames.push_back(cam2Frame);
        inputFrames.push_back(cam3Frame);
        inputFrames.push_back(cam4Frame);
        inputFrames.push_back(cam5Frame);
        inputFrames.push_back(cam6Frame);
        inputFrames.push_back(cam7Frame);
        inputFrames.push_back(cam8Frame);

        //get the warped and registered images
        outputFrames = getWarpedRegisteredImages(inputFrames);
        inputFrames.clear();

        //fetch the output registered frames
        cam1Frame = outputFrames.at(0);
        cam2Frame = outputFrames.at(1);
        cam3Frame = outputFrames.at(2);
        cam4Frame = outputFrames.at(3);
        cam5Frame = outputFrames.at(4);
        cam6Frame = outputFrames.at(5);
        cam7Frame = outputFrames.at(6);
        cam8Frame = outputFrames.at(7);
        outputFrames.clear();

        width = cam2Frame.size().width;
        height = cam2Frame.size().height;

        //cout << width << "," << height << "\n";
        Rect roi = Rect(150,10,1575,875);

	//imshow("cam1", cam1Frame(roi));
	//imshow("cam2", cam2Frame(roi));
	//imshow("cam3", cam3Frame(roi));
	//imshow("cam4", cam4Frame(roi));
	//imshow("cam5", cam5Frame(roi));
	//imshow("cam6", cam6Frame(roi));
	//imshow("cam7", cam7Frame(roi));
	//imshow("cam8", cam8Frame(roi));

        //vector to hold led coordinates
        vector<Point2f> cam1LedPos, cam2LedPos, cam3LedPos, cam4LedPos, cam5LedPos, cam6LedPos, cam7LedPos, cam8LedPos;

        //get led coordinates for each of the frame
        cam1LedPos = getLedCoordinates(cam1Frame(roi));
        cam2LedPos = getLedCoordinates(cam2Frame(roi));
        cam3LedPos = getLedCoordinates(cam3Frame(roi));
        cam4LedPos = getLedCoordinates(cam4Frame(roi));
        cam5LedPos = getLedCoordinates(cam5Frame(roi));
        cam6LedPos = getLedCoordinates(cam6Frame(roi));
        cam7LedPos = getLedCoordinates(cam7Frame(roi));
        cam8LedPos = getLedCoordinates(cam8Frame(roi));

        //function to get average LED position
        //averageLedPosition = getAveragePosition8cams(cam1LedPos, cam2LedPos, cam3LedPos, cam4LedPos, cam5LedPos, cam6LedPos, cam7LedPos, cam8LedPos);
        //averageRedLed = averageLedPosition.at(0);
        //averageGreenLed = averageLedPosition.at(1);

        //add x,y of red and green red to individual vectors
        //red_x.push_back((float)averageRedLed.x);
        //red_y.push_back((float)averageRedLed.y);
        //green_x.push_back((float)averageGreenLed.x);
        //green_y.push_back((float)averageGreenLed.y);

	cam1_red_x.push_back((float)cam1LedPos.at(0).x);
        cam1_red_y.push_back((float)cam1LedPos.at(0).y);
        cam1_green_x.push_back((float)cam1LedPos.at(1).x);
        cam1_green_y.push_back((float)cam1LedPos.at(1).y);

        cam2_red_x.push_back((float)cam2LedPos.at(0).x);
        cam2_red_y.push_back((float)cam2LedPos.at(0).y);
        cam2_green_x.push_back((float)cam2LedPos.at(1).x);
        cam2_green_y.push_back((float)cam2LedPos.at(1).y);

        cam3_red_x.push_back((float)cam3LedPos.at(0).x);
        cam3_red_y.push_back((float)cam3LedPos.at(0).y);
        cam3_green_x.push_back((float)cam3LedPos.at(1).x);
        cam3_green_y.push_back((float)cam3LedPos.at(1).y);

        cam4_red_x.push_back((float)cam4LedPos.at(0).x);
        cam4_red_y.push_back((float)cam4LedPos.at(0).y);
        cam4_green_x.push_back((float)cam4LedPos.at(1).x);
        cam4_green_y.push_back((float)cam4LedPos.at(1).y);

        cam5_red_x.push_back((float)cam5LedPos.at(0).x);
        cam5_red_y.push_back((float)cam5LedPos.at(0).y);
        cam5_green_x.push_back((float)cam5LedPos.at(1).x);
        cam5_green_y.push_back((float)cam5LedPos.at(1).y);

        cam6_red_x.push_back((float)cam6LedPos.at(0).x);
        cam6_red_y.push_back((float)cam6LedPos.at(0).y);
        cam6_green_x.push_back((float)cam6LedPos.at(1).x);
        cam6_green_y.push_back((float)cam6LedPos.at(1).y);

        cam7_red_x.push_back((float)cam7LedPos.at(0).x);
        cam7_red_y.push_back((float)cam7LedPos.at(0).y);
        cam7_green_x.push_back((float)cam7LedPos.at(1).x);
        cam7_green_y.push_back((float)cam7LedPos.at(1).y);

        cam8_red_x.push_back((float)cam8LedPos.at(0).x);
        cam8_red_y.push_back((float)cam8LedPos.at(0).y);
        cam8_green_x.push_back((float)cam8LedPos.at(1).x);
        cam8_green_y.push_back((float)cam8LedPos.at(1).y);

        //if escape is pressed, kill the window
        if((cvWaitKey(10) & 255) == 27) break;
    }

    //write Led position to csv file
    writeLedPosToFile8cams(cam1_red_x, cam1_red_y, cam1_green_x, cam1_green_y, cam2_red_x, cam2_red_y, cam2_green_x, cam2_green_y,
                           cam3_red_x, cam3_red_y, cam3_green_x, cam3_green_y, cam4_red_x, cam4_red_y, cam4_green_x, cam4_green_y,
                           cam5_red_x, cam5_red_y, cam5_green_x, cam5_green_y, cam6_red_x, cam6_red_y, cam6_green_x, cam6_green_y,
                           cam7_red_x, cam7_red_y, cam7_green_x, cam7_green_y, cam8_red_x, cam8_red_y, cam8_green_x, cam8_green_y,
                           width, height, outputFileName);
    LOGLN("Finished writing the position to csv file");

    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
