/**
 * @file SURF_descriptor
 * @brief SURF detector + descritpor + BruteForce Matcher + drawing matches with OpenCV functions
 * @author A. Huaman
 *
 * Modified to work with the current OpenCV (2.4) C++ bindings
 */

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;

void readme();

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { return -1; }

  Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_2 = imread( argv[2], CV_LOAD_IMAGE_GRAYSCALE );
  
  if( !img_1.data || !img_2.data )
  { return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian = 400;

  // SURFFeatureDetector is done this way now
  Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");

  std::vector<KeyPoint> keypoints_1, keypoints_2;

  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );

  //-- Step 2: Calculate descriptors (feature vectors)
  
  // SURFDescriptorExtractor is done this way now
  Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
  //SurfDescriptorExtractor extractor;

  Mat descriptors_1, descriptors_2;

  extractor->compute( img_1, keypoints_1, descriptors_1 );
  extractor->compute( img_2, keypoints_2, descriptors_2 );

  //-- Step 3: Matching descriptor vectors with a brute force matcher
  //BruteForceMatcher< L2<float> > matcher;
  // BruteForceMatcher now done as a Ptr<DescriptorMatcher>
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
  std::vector< DMatch > matches;
  matcher->match( descriptors_1, descriptors_2, matches );

  //-- Draw matches
  Mat img_matches;
  drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches ); 

  //-- Show detected matches
  imshow("Matches", img_matches );
  imwrite("result.png", img_matches );
  waitKey(0);

  return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
