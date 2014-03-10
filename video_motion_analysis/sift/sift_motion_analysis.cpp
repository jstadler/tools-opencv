// standard
#include <iostream>
#include <string>

// opencv
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"

#define M_PI 3.14159

using namespace std;

int end_frame = 200;
int start_frame = 0;
int every_n_frames = 3;
float sift_match_perc = 0.5;

cv::VideoCapture vidCap;
cv::Mat currentImage, prevImage, currGray, prevGray, flow, drawTo;
int frameCount;
cv::TermCriteria termcrit;
bool needToInit;
float rel_vec_x, rel_vec_y;

void shutItDown( int e ){
	cout << "Press any key to close" << endl;
	getchar();
	exit( e );
}

///////////////////////////////////////////////////////////////////////////////
// initalize the video for processing
///////////////////////////////////////////////////////////////////////////////
void initVideo( string videoFile ){

	vidCap = cv::VideoCapture( videoFile );

	if( !vidCap.isOpened() ){
		cout << "Video did not open" << endl;
		shutItDown(-1);
	}
	cout << "width: " << vidCap.get(CV_CAP_PROP_FRAME_WIDTH) << endl;
	cout << "height: " << vidCap.get(CV_CAP_PROP_FRAME_HEIGHT) << endl;

	currentImage = cv::Mat(640,480, CV_8UC3);
	frameCount = 0;
	vidCap.read(currentImage);
	cv::cvtColor(currentImage, currGray, CV_BGR2GRAY);
	swap(prevGray, currGray); swap(prevImage, currentImage);
	flow = cv::Mat(currentImage.size(), CV_32FC2);

	termcrit = cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
	needToInit = true;

	rel_vec_x = 1.0f;
	rel_vec_y = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////
// close the video
///////////////////////////////////////////////////////////////////////////////
void stopVideo(){
	vidCap.~VideoCapture();
}

float euclideanDist(cv::KeyPoint& p, cv::KeyPoint& q) {

	float diffx = p.pt.x - q.pt.x;
	float diffy = p.pt.y - q.pt.y;

    return std::sqrt((float)(diffx*diffx + diffy*diffy));
}

void sift_optical_flow(){
	vidCap.read( currentImage );
	cv::cvtColor(currentImage, currGray, CV_BGR2GRAY);
	swap(prevGray, currGray);
	swap(prevImage, currentImage);

	cv::SiftFeatureDetector detector;
	std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
	cv::Mat descriptors_1, descriptors_2;
	detector.detect( prevGray, keypoints_2 );
	cv::SiftDescriptorExtractor extractor;
	extractor.compute( prevGray, keypoints_2, descriptors_2);
	cv::BFMatcher matcher(cv::NORM_L2, false);
	std::vector<cv::DMatch> matches;

	int count = 0;
	// loop through video
	while( vidCap.read( currentImage ) && count < end_frame ){
		cv::cvtColor(currentImage, currGray, CV_BGR2GRAY);
		count++;
		if( count < start_frame ) continue;

		if( count % every_n_frames != 0 ) continue;

		//printf("frame: %d/%d\n", count, end_frame);

		detector.detect( currGray, keypoints_1);
		extractor.compute( currGray, keypoints_1, descriptors_1);

		// get sift vectors

		//printf("1: %dx%d\n", descriptors_1.rows, descriptors_1.cols );
		//printf("2: %dx%d\n", descriptors_2.rows, descriptors_2.cols );

		// matching descriptors
		//printf("d1: %d d2: %d\n", descriptors_1.size(), descriptors_2.size());
		
		// returns nearest neighbor for each element of descriptors_1
		matcher.match(descriptors_1, descriptors_2, matches);

		//printf("matches: %d d1: %d d2: %d\n", matches.size(), descriptors_1.size(), descriptors_2.size());
		//printf("match distance: %f\n", matches[0].distance );
		std::sort(matches.begin(), matches.end());
		
		//printf("%f vs. %f \n", matches[0].distance, matches[matches.size()-1].distance);

		//printf("sift match perc: %f\n", sift_match_perc);
		matches.resize((int)(matches.size()*sift_match_perc));

		std::vector<cv::DMatch> matches2;

		//filter_matches( &matches, keypoints_1, keypoints_2 );
		float ed;
		cv::DMatch current;
		for(int i = 0; i < matches.size(); i++){
			current = matches.at(i);
			ed = euclideanDist(keypoints_1[current.queryIdx], keypoints_2[current.trainIdx]);
			if (ed > 2.0f  && ed < 10.0f) {
				matches2.push_back(current);
			}
		}

		// copy to 
		currentImage.copyTo( drawTo );
		for( int j = 0; j < matches2.size(); j++ ){
			cv::line(drawTo, keypoints_1[matches2[j].queryIdx].pt, keypoints_2[matches2[j].trainIdx].pt, cv::Scalar(0,0,255),1,8,0);
			
			// draw the head
			cv::circle(drawTo, keypoints_1[matches2[j].queryIdx].pt, 1, cv::Scalar(0,255,0), -1,8,0);

			float x1 = keypoints_1[matches2[j].queryIdx].pt.x;
			float y1 = keypoints_1[matches2[j].queryIdx].pt.y;

			float x2 = keypoints_2[matches2[j].trainIdx].pt.x;
			float y2 = keypoints_2[matches2[j].trainIdx].pt.y;

			float xx = x2 - x1;
			float yy = y2 - y1;

			float xcom = (xx)/2. + x1;
			float ycom = (yy)/2. + y1;

			float vec_length = sqrt( (xx*xx) + (yy*yy) );

			float val = yy/xx;
			float deg = atan(val) * 180 / M_PI;
			float deg2 = deg;
			// quad 3 - +180deg
			if( xx < 0.0f && yy < 0.0f){
				deg2 += 180;
			// quad 2 - +180deg
			}else if( xx < 0.0f){
				deg2 += 180;
			// quad 4 - +360deg
			}else if(yy < 0.0f){
				deg2 += 360;
			}

			// draw the COM
			cv::circle(drawTo, cv::Point2f(xcom, ycom), 1, cv::Scalar(255,255,0), -1,8,0);
		}

		
		//printf("m2 size: %d\n", matches2.size() );

		cv::imshow("matches", drawTo);
		cv::waitKey(1);

		swap(keypoints_1, keypoints_2);		swap(descriptors_1, descriptors_2);
		matches.clear();
	}
}



int main ( int argc, char *argv[] ){
	if( argc != 2 ){
		cout << "Program requires a single argument," << endl <<
			"which is the location of a video file." << endl << endl;
		shutItDown(-1);
	}

	string file = argv[1];

	initVideo( file );

	sift_optical_flow();

	stopVideo();
	getchar();
}