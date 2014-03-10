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
float min_vector_length = 0.0f;

std::vector<cv::Point2f> points[2];

int width = 640;
int height = 480;

// Polar representations of directions
struct Pole_Node{
	float x;
	float y;
	float r;
	float theta;
	int timestamp;
	Pole_Node * next;
	Pole_Node():
	x(-1){}
};

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
	width = vidCap.get(CV_CAP_PROP_FRAME_WIDTH);
	height = vidCap.get(CV_CAP_PROP_FRAME_HEIGHT);
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

///////////////////////////////////////////////////////////////////////////////
//  returns the distance between two Point2f's
///////////////////////////////////////////////////////////////////////////////
float distanceBetweenPoints( cv::Point2f p1, cv::Point2f p2 ){
	float xx = p2.x - p1.x;
	float yy = p2.y - p1.y;
	return sqrt( xx*xx + yy*yy );
}

///////////////////////////////////////////////////////////////////////////////
// Generate polar coordinates from a processed video
///////////////////////////////////////////////////////////////////////////////
void sparse_optical_flow(){
	int count = 0;

	// loop through video
	while( vidCap.read( currentImage ) && count < end_frame ){
		count++;
		if( count < start_frame ) continue;
		if( count % every_n_frames != 0 ) continue;
		cout << "frame: " << count << "/" << end_frame << endl;

		cv::cvtColor( currentImage, currGray, CV_BGR2GRAY );
		currentImage.copyTo( drawTo );
		if( needToInit ) {
			goodFeaturesToTrack( currGray, points[1], 10000, 0.01, 3, cv::Mat(), 3, 0, 0.04 );
			cornerSubPix( currGray, points[1], cv::Size(10,10), cv::Size(-1, -1), termcrit );
			needToInit = false;
		}else if( !points[0].empty() ){
			vector<uchar> status;
			vector<float> err;
			if(prevGray.empty()){
					currGray.copyTo(prevGray);
			}
			calcOpticalFlowPyrLK( prevGray, currGray, points[0], points[1], status, err, cv::Size(5,5), 3, termcrit, 0, 0.001);
			size_t i, k;
			for( i = k = 0; i < points[1].size(); i++){
				if(!status[i]){
					continue;
				}

				points[1][k++] = points[1][i];

				float dist = distanceBetweenPoints( points[0][i], points[1][i] );

				if( dist > min_vector_length && dist < 10.00f ){

					// remove negative values
					if(points[0][i].x < 0.0f){
						points[0][i].x = 0.0f;
					}
					if(points[0][i].y < 0.0f){
						points[0][i].y = 0.0f;
					}
					if(points[1][i].x < 0.0f){
						points[1][i].x = 0.0f;
					}
					if(points[1][i].y < 0.0f){
						points[1][i].y = 0.0f;
					}

					// remove high values
					if(points[0][i].x >= width){
						points[0][i].x = width - 0.0001f;
					}
					if(points[0][i].y >= height){
						points[0][i].y = height - 0.0001f;
					}
					if(points[1][i].x >= width){
						points[1][i].x = width - 0.0001f;
					}
					if(points[1][i].y >= height){
						points[1][i].y = height - 0.0001f;
					}

					// dont use vectors that start and end in the same position
					if( (points[0][i].x == points[1][i].x) && (points[0][i].y == points[1][i].y) ){
						continue;
					}

					// draw vector
					cv::line(drawTo, points[0][i], points[1][i], cv::Scalar(0,0,255),1,8,0);
					// draw the head
					cv::circle(drawTo, points[1][i], 1, cv::Scalar(0,255,0), -1,8,0);
				}
			}
			points[1].resize(k);
		}
		cv::imshow("vectors", drawTo);
		cv::waitKey(1);

		swap(points[1], points[0]);
		swap(prevGray, currGray);
		swap(prevImage, currentImage);
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

	sparse_optical_flow();

	stopVideo();
	getchar();
}