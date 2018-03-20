#include"stitchingUtils.hpp"

using namespace cv;
using namespace cv::detail;
using namespace std;

// initialize variables
double work_megapix = 0.6;
double compose_megapix = -1;
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
string warp_type = "plane";

/**function to get warped and transformed image
INPUT: input image, mask, corner point, roi, output image mask reference, output image reference
**/
void registerImages(InputArray _img, InputArray _mask, Point tl, Rect dst_roi_, Mat dst_mask_, Mat dst_)
{
    int corner_x, corner_y;
    Mat img = _img.getMat();
    Mat mask = _mask.getMat();

    CV_Assert(img.type() == CV_16SC3);
    CV_Assert(mask.type() == CV_8U);

    //update the corner points for each image
    corner_x = tl.x - dst_roi_.x;
    corner_y = tl.y - dst_roi_.y;

    //update output image and image mask with corners updates and put it in global world system
    for (int y = 0; y < img.rows; ++y)
    {
        const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
        Point3_<short> *dst_row = dst_.ptr<Point3_<short> >(corner_y + y);
        const uchar *mask_row = mask.ptr<uchar>(y);
        uchar *dst_mask_row = dst_mask_.ptr<uchar>(corner_y + y);
        
        //cout << corner_y + y << "," << corner_y << "," << y << "\n";

        for (int x = 0; x < img.cols; ++x)
        {
            if (mask_row[x])
                dst_row[corner_x + x] = src_row[x];
            dst_mask_row[corner_x + x] |= mask_row[x];
            //cout << corner_x + x << "\n";
        }
    }
}


//function to get warped Registered image on which we can apply position tracking
vector<Mat> getWarpedRegisteredImages(vector<Mat> InputImage)
{
    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_compose_scale_set = false;

    Mat full_img, img;
    int num_images = (int)InputImage.size();
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;
    vector<CameraParams> cameras(num_images);

    //resize each image to suit the function calls
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
        resize(full_img, img, Size(), seam_scale, seam_scale);
        images[i] = img.clone();
    }
    full_img.release();
    img.release();

    //load camera parameters from the yml file
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

    //panorama scale estimation
    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    //perform wave correction
    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    //Warping images (auxiliary)... ;

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

    //create the rotation warper object
    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    //warp the images and mask
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

    //Finished warping images

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    double compose_work_aspect = 1;
    vector<Mat> RegisteredImages;

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

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        Rect dst_roi, dst_roi_;
        //create blank image and image mask which will store the warped image
        dst_roi = resultRoi(corners, sizes);
        Mat dst_, dst_mask_;
        dst_.create(dst_roi.size(), CV_16SC3);
        dst_.setTo(Scalar::all(0));
        dst_mask_.create(dst_roi.size(), CV_8U);
        dst_mask_.setTo(Scalar::all(0));
        dst_roi_ = dst_roi;

        //Transforming Images
        //save warped and Transformed image for each camera and also update the new corner
        registerImages(img_warped_s, mask_warped, corners[img_idx], dst_roi_, dst_mask_, dst_);
        //LOGLN("Done loading registered images");

        //NOTE: covert the image to 8U so as to resolve the issue with display.. some sort of bug in opencv
        dst_.convertTo(dst_, CV_8U);
        RegisteredImages.push_back(dst_);
    }
    return RegisteredImages;
}

vector<Point2f> getLedCoordinates(Mat frame)
{
    Point2f redLedPos = Point2f(-1,-1);
    Point2f greenLedPos = Point2f(-1,-1);
    vector<Point2f> ledPos;
    Mat thresholdedImage;

    //thresholded image
    threshold(frame, thresholdedImage, 100, 255,THRESH_BINARY);

    //remove small noise from the red and green colro thesholded image
    Mat str_el = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2,2));
    morphologyEx(thresholdedImage, thresholdedImage, cv::MORPH_OPEN, str_el);
    morphologyEx(thresholdedImage, thresholdedImage, cv::MORPH_CLOSE, str_el);

    // Convert input image to HSV
    Mat hsv_image;
    cvtColor(thresholdedImage, hsv_image, cv::COLOR_BGR2HSV);

    // Threshold the HSV image, keep only the red pixels
    Mat lower_red_hue_range, upper_red_hue_range;
    inRange(hsv_image, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), lower_red_hue_range);
    inRange(hsv_image, cv::Scalar(160, 100, 100), cv::Scalar(179, 255, 255), upper_red_hue_range);

    // Combine the above two image
    Mat red_hue_image;
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, red_hue_image);
    //blur the image to avoid false positives
    GaussianBlur(red_hue_image, red_hue_image, cv::Size(9, 9), 2, 2);

    // Threshold the HSV image, keep only the green pixels
    Mat green_hue_image;
    inRange(hsv_image, cv::Scalar(50, 50, 120), cv::Scalar(70, 255, 255), green_hue_image);

    //blur the image to avoid false positives
    GaussianBlur(green_hue_image, green_hue_image, cv::Size(9, 9), 2, 2);

    //find center of red contours and green contours with max area
    vector<vector<Point> > redContours, greenContours;
    vector<Vec4i> redHeirarchy, greenHeirarchy;
    findContours(red_hue_image.clone(), redContours, redHeirarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    findContours(green_hue_image.clone(), greenContours, greenHeirarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    //iterate through each contour and find the centroid of max area contour for red LED
    double largest_area = 0;
    int largest_contour_index = -1;
    size_t count = (int)redContours.size();
    if(count>0){
        for(unsigned int i = 0; i< count; i++ )
        {
            //  Find the area of contour
            double a=contourArea(redContours[i],false);
            if(a>largest_area & a<200){
                largest_area=a;
                // Store the index of largest contour
                largest_contour_index=i;
        	}
        }
	if (largest_contour_index !=-1){
        	Moments redMoment = moments(redContours[largest_contour_index], false);
        	//get center of all red led positions crossing threshold
        	redLedPos = Point2f(redMoment.m10/redMoment.m00, redMoment.m01/redMoment.m00);
	}
    }

    //iterate through each contour and find the centroid of max area contour for green led
    largest_area = 0;
    largest_contour_index = -1;
    count = (int)greenContours.size();
    if(count>0){
        //iterate through each contour and find the centroid of max area contour
        for( unsigned int i = 0; i< count; i++ )
        {
        //  Find the area of contour
        double a=contourArea(greenContours[i],false);
        if(a>largest_area & a<200){
            largest_area=a;
            // Store the index of largest contour
            largest_contour_index=i;
            }
        }
	if (largest_contour_index !=-1){
        	Moments greenMoment = moments(greenContours[largest_contour_index], false);
        	//get center of all green led positions crossing threshold
        	greenLedPos = Point2f(greenMoment.m10/greenMoment.m00, greenMoment.m01/greenMoment.m00);
	}
    }

    //draw circle around red and green Led
    //circle(frame, redLedPos, 2, Scalar(0,0,255), 2);
    //circle(frame, greenLedPos, 2, Scalar(0,255,0), 2);
    //show frame with colored LED
    //imshow("Frame", frame);
    //show the thresholded frame
    //imshow("Thresholded Frame", thresholdedImage);

    //add red and green led position to vector and return it
    ledPos.push_back(redLedPos);
    ledPos.push_back(greenLedPos);

    return ledPos;
}

//function to rotate the image
void rotate_90n(cv::Mat const &src, cv::Mat &dst, int angle)
{
     CV_Assert(angle % 90 == 0 && angle <= 360 && angle >= -360);
     if(angle == 270 || angle == -90){
        // Rotate clockwise 270 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 0);
     }
     else if(angle == 180 || angle == -180){
        // Rotate clockwise 180 degrees
        cv::flip(src, dst, -1);
     }
     else if(angle == 90 || angle == -270){
        // Rotate clockwise 90 degrees
        cv::transpose(src, dst);
        cv::flip(dst, dst, 1);
     }
     else if(angle == 360 || angle == 0 || angle == -360){
        if(src.data != dst.data){
            src.copyTo(dst);
        }
     }
}

//function to get average position estimate for 8 cameras
vector<Point2f> getAveragePosition8cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos, vector<Point2f> &cam4LedPos, 
	vector<Point2f> &cam5LedPos, vector<Point2f> &cam6LedPos, vector<Point2f> &cam7LedPos, vector<Point2f> &cam8LedPos)
{
    //initialize the variables for average Led positions
    Point2f averageRedLed, averageGreenLed;
    vector<Point2f> redLedPos, greenLedPos;
    vector<Point2f> averageLedPosition;

    //iterate over each camera and add its position to single vector which is later used
    //for average position calculation only if it is not (-1.-1)
    if (cam1LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam1LedPos.at(0));
    }
    if (cam1LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam1LedPos.at(1));
    }
    if (cam2LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam2LedPos.at(0));
    }
    if (cam2LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam2LedPos.at(1));
    }
    if (cam3LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam3LedPos.at(0));
    }
    if (cam3LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam3LedPos.at(1));
    }
    if (cam4LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam4LedPos.at(0));
    }
    if (cam4LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam4LedPos.at(1));
    }
    if (cam5LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam5LedPos.at(0));
    }
    if (cam5LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam5LedPos.at(1));
    }
    if (cam6LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam6LedPos.at(0));
    }
    if (cam6LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam6LedPos.at(1));
    }
    if (cam7LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam7LedPos.at(0));
    }
    if (cam7LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam7LedPos.at(1));
    }
    if (cam8LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam8LedPos.at(0));
    }
    if (cam8LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam8LedPos.at(1));
    }

    // if vector from each camera is empty assign average position to be (-1,-1)
    //else calculate average of all position in vector for both red and green Led
    if(redLedPos.empty())
    {
        averageRedLed = Point2f(-1,-1);
    }
    else{
        //calculate average red led position
        Point2f meanPos;
        for(unsigned int i=0; i<redLedPos.size();++i){
            meanPos.x = meanPos.x + redLedPos.at(i).x;
            meanPos.y = meanPos.y + redLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/redLedPos.size();
        float meanPos_y = (float)meanPos.y/redLedPos.size();
        averageRedLed = Point2f(meanPos_x, meanPos_y);
    }

    if(greenLedPos.empty())
    {
        averageGreenLed = Point2f(-1,-1);
    }
    else{
        //calculate average green led position
        Point2f meanPos;
        for(unsigned int i=0; i<greenLedPos.size();++i){
            meanPos.x = meanPos.x + greenLedPos.at(i).x;
            meanPos.y = meanPos.y + greenLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/greenLedPos.size();
        float meanPos_y = (float)meanPos.y/greenLedPos.size();
        averageGreenLed = Point2f(meanPos_x, meanPos_y);
    }

    //add average red and green led position to a vector
    averageLedPosition.push_back(averageRedLed);
    averageLedPosition.push_back(averageGreenLed);

    //return the vector
    return averageLedPosition;
}

//function to get average position estimate for 4 cameras
vector<Point2f> getAveragePosition4cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos, vector<Point2f> &cam4LedPos)
{
    //initialize the variables for average Led positions
    Point2f averageRedLed, averageGreenLed;
    vector<Point2f> redLedPos, greenLedPos;
    vector<Point2f> averageLedPosition;

    //iterate over each camera and add its position to single vector which is later used
    //for average position calculation only if it is not (-1.-1)
        if (cam1LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam1LedPos.at(0));
    }
    if (cam1LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam1LedPos.at(1));
    }
    if (cam2LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam2LedPos.at(0));
    }
    if (cam2LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam2LedPos.at(1));
    }
    if (cam3LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam3LedPos.at(0));
    }
    if (cam3LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam3LedPos.at(1));
    }
    if (cam4LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam4LedPos.at(0));
    }
    if (cam4LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam4LedPos.at(1));
    }

    // if vector from each camera is empty assign average position to be (-1,-1)
    //else calculate average of all position in vector for both red and green Led
    if(redLedPos.empty())
    {
        averageRedLed = Point2f(-1,-1);
    }
    else{
        //calculate average red led position
        Point2f meanPos;
        for(unsigned int i=0; i<redLedPos.size();++i){
            meanPos.x = meanPos.x + redLedPos.at(i).x;
            meanPos.y = meanPos.y + redLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/redLedPos.size();
        float meanPos_y = (float)meanPos.y/redLedPos.size();
        averageRedLed = Point2f(meanPos_x, meanPos_y);
    }

    if(greenLedPos.empty())
    {
        averageGreenLed = Point2f(-1,-1);
    }
    else{
        //calculate average green led position
        Point2f meanPos;
        for(unsigned int i=0; i<greenLedPos.size();++i){
            meanPos.x = meanPos.x + greenLedPos.at(i).x;
            meanPos.y = meanPos.y + greenLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/greenLedPos.size();
        float meanPos_y = (float)meanPos.y/greenLedPos.size();
        averageGreenLed = Point2f(meanPos_x, meanPos_y);
    }

    //add average red and green led position to a vector
    averageLedPosition.push_back(averageRedLed);
    averageLedPosition.push_back(averageGreenLed);

    //return the vector
    return averageLedPosition;
}

//function to get average position estimate for 3 cameras
vector<Point2f> getAveragePosition3cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos, vector<Point2f> &cam3LedPos)
{
    //initialize the variables for average Led positions
    Point2f averageRedLed, averageGreenLed;
    vector<Point2f> redLedPos, greenLedPos;
    vector<Point2f> averageLedPosition;

    //iterate over each camera and add its position to single vector which is later used
    //for average position calculation only if it is not (-1.-1)
    if (cam1LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam1LedPos.at(0));
    }
    if (cam1LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam1LedPos.at(1));
    }
    if (cam2LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam2LedPos.at(0));
    }
    if (cam2LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam2LedPos.at(1));
    }
    if (cam3LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam3LedPos.at(0));
    }
    if (cam3LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam3LedPos.at(1));
    }

    // if vector from each camera is empty assign average position to be (-1,-1)
    //else calculate average of all position in vector for both red and green Led
    if(redLedPos.empty())
    {
        averageRedLed = Point2f(-1,-1);
    }
    else{
        //calculate average red led position
        Point2f meanPos;
        for(unsigned int i=0; i<redLedPos.size();++i){
            meanPos.x = meanPos.x + redLedPos.at(i).x;
            meanPos.y = meanPos.y + redLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/redLedPos.size();
        float meanPos_y = (float)meanPos.y/redLedPos.size();
        averageRedLed = Point2f(meanPos_x, meanPos_y);
    }

    if(greenLedPos.empty())
    {
        averageGreenLed = Point2f(-1,-1);
    }
    else{
        //calculate average green led position
        Point2f meanPos;
        for(unsigned int i=0; i<greenLedPos.size();++i){
            meanPos.x = meanPos.x + greenLedPos.at(i).x;
            meanPos.y = meanPos.y + greenLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/greenLedPos.size();
        float meanPos_y = (float)meanPos.y/greenLedPos.size();
        averageGreenLed = Point2f(meanPos_x, meanPos_y);
    }

    //add average red and green led position to a vector
    averageLedPosition.push_back(averageRedLed);
    averageLedPosition.push_back(averageGreenLed);

    //return the vector
    return averageLedPosition;
}


//function to get average position estimate for 2 cameras
vector<Point2f> getAveragePosition2cams(vector<Point2f> &cam1LedPos, vector<Point2f> &cam2LedPos)
{
    //initialize the variables for average Led positions
    Point2f averageRedLed, averageGreenLed;
    vector<Point2f> redLedPos, greenLedPos;
    vector<Point2f> averageLedPosition;

    //iterate over each camera and add its position to single vector which is later used
    //for average position calculation only if it is not (-1.-1)
    if (cam1LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam1LedPos.at(0));
    }
    if (cam1LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam1LedPos.at(1));
    }
    if (cam2LedPos.at(0) != Point2f(-1,-1)){
        redLedPos.push_back(cam2LedPos.at(0));
    }
    if (cam2LedPos.at(1) != Point2f(-1,-1)){
        greenLedPos.push_back(cam2LedPos.at(1));
    }

    // if vector from each camera is empty assign average position to be (-1,-1)
    //else calculate average of all position in vector for both red and green Led
    if(redLedPos.empty())
    {
        averageRedLed = Point2f(-1,-1);
    }
    else{
        //calculate average red led position
        Point2f meanPos;
        for(unsigned int i=0; i<redLedPos.size();++i){
            meanPos.x = meanPos.x + redLedPos.at(i).x;
            meanPos.y = meanPos.y + redLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/redLedPos.size();
        float meanPos_y = (float)meanPos.y/redLedPos.size();
        averageRedLed = Point2f(meanPos_x, meanPos_y);
    }

    if(greenLedPos.empty())
    {
        averageGreenLed = Point2f(-1,-1);
    }
    else{
        //calculate average green led position
        Point2f meanPos;
        for(unsigned int i=0; i<greenLedPos.size();++i){
            meanPos.x = meanPos.x + greenLedPos.at(i).x;
            meanPos.y = meanPos.y + greenLedPos.at(i).y;
        }
        float meanPos_x = (float)meanPos.x/greenLedPos.size();
        float meanPos_y = (float)meanPos.y/greenLedPos.size();
        averageGreenLed = Point2f(meanPos_x, meanPos_y);
    }

    //add average red and green led position to a vector
    averageLedPosition.push_back(averageRedLed);
    averageLedPosition.push_back(averageGreenLed);

    //return the vector
    return averageLedPosition;
}

//function to write Led position to csv file
void writeLedPosToFile(vector<float> red_x, vector<float> red_y, vector<float> green_x, vector<float> green_y, int width, int height, const char* filename){
    ofstream fout(filename);
    //write width and height to file
    fout << width << "," << height << "\n";
    fout << "redX" << "," << "redY" << "," << "greenX" << "," << "greenY" << "\n";
    for(unsigned int i=0; i< red_x.size(); ++i){
        fout << red_x.at(i) << "," << red_y.at(i) << "," << green_x.at(i) << "," << green_y.at(i) << "\n";
    }
    fout.close();
}

//function to write Led position of 4cams to csv file
void writeLedPosToFile4cams(vector<float> cam1_red_x, vector<float> cam1_red_y, vector<float> cam1_green_x, vector<float> cam1_green_y,
                            vector<float> cam2_red_x, vector<float> cam2_red_y, vector<float> cam2_green_x, vector<float> cam2_green_y,
                            vector<float> cam3_red_x, vector<float> cam3_red_y, vector<float> cam3_green_x, vector<float> cam3_green_y,
                            vector<float> cam4_red_x, vector<float> cam4_red_y, vector<float> cam4_green_x, vector<float> cam4_green_y,
                            int width, int height, const char* filename){
    ofstream fout(filename);
    //write width and height to file
    fout << width << "," << height << "\n";
    fout << "cam1_redX" << "," << "cam1_redY" << "," << "cam1_greenX" << "," << "cam1_greenY" << ","
         << "cam2_redX" << "," << "cam2_redY" << "," << "cam2_greenX" << "," << "cam2_greenY" << ","
         << "cam3_redX" << "," << "cam3_redY" << "," << "cam3_greenX" << "," << "cam3_greenY" << ","
         << "cam4_redX" << "," << "cam4_redY" << "," << "cam4_greenX" << "," << "cam4_greenY" << "\n";
    for(unsigned int i=0; i< cam1_red_x.size(); ++i){
        fout << cam1_red_x.at(i) << "," << cam1_red_y.at(i) << "," << cam1_green_x.at(i) << "," << cam1_green_y.at(i) << ","
             << cam2_red_x.at(i) << "," << cam2_red_y.at(i) << "," << cam2_green_x.at(i) << "," << cam2_green_y.at(i) << ","
             << cam3_red_x.at(i) << "," << cam3_red_y.at(i) << "," << cam3_green_x.at(i) << "," << cam3_green_y.at(i) << ","
             << cam4_red_x.at(i) << "," << cam4_red_y.at(i) << "," << cam4_green_x.at(i) << "," << cam4_green_y.at(i) << "\n";
    }
    fout.close();
}

//function to write Led position of 8cams to csv file
void writeLedPosToFile8cams(vector<float> cam1_red_x, vector<float> cam1_red_y, vector<float> cam1_green_x, vector<float> cam1_green_y,
                            vector<float> cam2_red_x, vector<float> cam2_red_y, vector<float> cam2_green_x, vector<float> cam2_green_y,
                            vector<float> cam3_red_x, vector<float> cam3_red_y, vector<float> cam3_green_x, vector<float> cam3_green_y,
                            vector<float> cam4_red_x, vector<float> cam4_red_y, vector<float> cam4_green_x, vector<float> cam4_green_y,
                            vector<float> cam5_red_x, vector<float> cam5_red_y, vector<float> cam5_green_x, vector<float> cam5_green_y,
                            vector<float> cam6_red_x, vector<float> cam6_red_y, vector<float> cam6_green_x, vector<float> cam6_green_y,
                            vector<float> cam7_red_x, vector<float> cam7_red_y, vector<float> cam7_green_x, vector<float> cam7_green_y,
                            vector<float> cam8_red_x, vector<float> cam8_red_y, vector<float> cam8_green_x, vector<float> cam8_green_y,
                            int width, int height, const char* filename){
    ofstream fout(filename);
    //write width and height to file
    fout << width << "," << height << "\n";
    fout << "cam1_redX" << "," << "cam1_redY" << "," << "cam1_greenX" << "," << "cam1_greenY" << ","
         << "cam2_redX" << "," << "cam2_redY" << "," << "cam2_greenX" << "," << "cam2_greenY" << ","
         << "cam3_redX" << "," << "cam3_redY" << "," << "cam3_greenX" << "," << "cam3_greenY" << ","
         << "cam4_redX" << "," << "cam4_redY" << "," << "cam4_greenX" << "," << "cam4_greenY" << ","
         << "cam5_redX" << "," << "cam5_redY" << "," << "cam5_greenX" << "," << "cam5_greenY" << ","
         << "cam6_redX" << "," << "cam6_redY" << "," << "cam6_greenX" << "," << "cam6_greenY" << ","
         << "cam7_redX" << "," << "cam7_redY" << "," << "cam7_greenX" << "," << "cam7_greenY" << ","
         << "cam8_redX" << "," << "cam8_redY" << "," << "cam8_greenX" << "," << "cam8_greenY" << "\n";
    for(unsigned int i=0; i< cam1_red_x.size(); ++i){
        fout << cam1_red_x.at(i) << "," << cam1_red_y.at(i) << "," << cam1_green_x.at(i) << "," << cam1_green_y.at(i) << ","
             << cam2_red_x.at(i) << "," << cam2_red_y.at(i) << "," << cam2_green_x.at(i) << "," << cam2_green_y.at(i) << ","
             << cam3_red_x.at(i) << "," << cam3_red_y.at(i) << "," << cam3_green_x.at(i) << "," << cam3_green_y.at(i) << ","
             << cam4_red_x.at(i) << "," << cam4_red_y.at(i) << "," << cam4_green_x.at(i) << "," << cam4_green_y.at(i) << ","
             << cam5_red_x.at(i) << "," << cam5_red_y.at(i) << "," << cam5_green_x.at(i) << "," << cam5_green_y.at(i) << ","
             << cam6_red_x.at(i) << "," << cam6_red_y.at(i) << "," << cam6_green_x.at(i) << "," << cam6_green_y.at(i) << ","
             << cam7_red_x.at(i) << "," << cam7_red_y.at(i) << "," << cam7_green_x.at(i) << "," << cam7_green_y.at(i) << ","
             << cam8_red_x.at(i) << "," << cam8_red_y.at(i) << "," << cam8_green_x.at(i) << "," << cam8_green_y.at(i) << "\n";
    }
    fout.close();
}
