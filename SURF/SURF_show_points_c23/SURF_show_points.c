/*
 * Jordan Stadler
 * April 2012
 *
 * show_SURF simply shows the SURF points found in an image or 
 * frame of a video
 *
 */

#include "cv.h" 
#include "highgui.h" 
#include <stdio.h> 
#include <stdlib.h>

#define THRESHOLD 500.00

// drawPoints loops through keyPoints and draws the points onto image using
void drawPoints( IplImage* image, CvSeq* keyPoints) {
	int k;
	for( k = 0; k < keyPoints->total; k++) {
		CvSURFPoint* r1 = (CvSURFPoint*)cvGetSeqElem( keyPoints,
				k );
		cvCircle( image, cvPointFrom32f( r1->pt), 1.0,
				  cvScalar(0, 0, 255, 0.0), -1, 8, 0);
	}
}

int main( int argc, char** argv ) {
	IplImage* bgr;
	IplImage* gray;

	if(argc < 2)
	{
		printf("Program Use: ./a.out <Image Location>\n");
		exit(0);
	}

	// RGB
	bgr = cvLoadImage( argv[1], 1);
		
	// Gray generation
	gray = cvCreateImage(cvGetSize(bgr), 8, 1);
	cvCvtColor(bgr, gray, CV_BGR2GRAY);

	// storage stores keypoints and descriptors during SURF extraction
	CvMemStorage* storage = cvCreateMemStorage(0);

	// CvSeq's to hold the keypoints and descriptors
	CvSeq* keypoints = 0, *descriptors = 0;

	// SURF parameters
	CvSURFParams params = cvSURFParams(THRESHOLD, 1);

	cvExtractSURF(gray, 0, &keypoints,
		&descriptors, storage, params, 0);

	drawPoints( bgr, keypoints);

	printf("Points found : %d\n", keypoints->total);

	while(1) {
		cvShowImage( "SURF points", bgr);
		int c;
		c = cvWaitKey(10);
		if( c == 27 || c == 113 ) {
			break;
		}
	}
	return 0;
}