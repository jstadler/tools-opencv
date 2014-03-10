/*
 * Jordan Stadler
 * April 2012
 *
 * show_SURF simply shows the SURF points found in an image or 
 * frame of a video
 *
 */

#include <stdio.h>
#include <iostream>
#include "cv.h" 
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

// needed for settign threshold on SURF 
#include "opencv2/nonfree/nonfree.hpp"

#define THRESHOLD 500.00

using namespace cv;

void drawPoints( Mat bgrImage, std::vector<KeyPoint> kp, int r, int g, int b) {
	for( int k = 0; k < kp.size(); k++) {
		circle(bgrImage, kp[k].pt, 1, Scalar(0, 0, 255), 1, 8, 0);
	}
}

int main( int argc, char** argv ) {
	Mat bgr;
	Mat gray;
	std::vector<KeyPoint> keypoints;

	if(argc < 2)
	{
		printf("Program Use: ./a.out <Image Location>\n");
		exit(0);
	}

	// BGR
	bgr = imread( argv[1], 1);
		
	// Gray generation
	cvtColor(bgr, gray, CV_RGB2GRAY, 0);

	// generates a detector and detects keypoints, no descriptors are extracted
	
	// Method without a threshold
	//Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	//detector->detect( gray, keypoints );

	// Threshold method
	SURF surfExtractor(THRESHOLD, 4, 2, false, false);
	surfExtractor.operator()(bgr, noArray(), keypoints);

	drawPoints( bgr, keypoints, 255, 0, 0);

	printf("Points found : %d\n", keypoints.size());

	while(1) {
		imshow( "SURF points", bgr);
		int c;
		c = cvWaitKey(10);
		if( c == 27 || c == 113 ) {
			break;
		}
	}
	return 0;
}