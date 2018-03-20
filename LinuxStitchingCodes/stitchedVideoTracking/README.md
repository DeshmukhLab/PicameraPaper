Compile using:

g++ -std=c++11 StitchedPositionTracking8cam.cpp stitchingUtils.cpp -o StitchedPositionTracking8cam `pkg-config --cflags --libs opencv`

Run using:

./StitchedPositionTracking8cam vid1.h264 vid2.h264 vid3.h264 vid4.h264....