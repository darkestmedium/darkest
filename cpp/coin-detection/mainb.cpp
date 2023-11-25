// System includes
#include <vector>
#include <iostream>
#include <fstream>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>




// Using namespace to nullify use of cv::function(); syntax
using namespace cv;
using namespace std;




Mat displayConnectedComponents(Mat &img) {
 // Make a copy of the image
 Mat imLabels = img.clone();

 // First let's find the min and max values in imLabels
 Point minLoc, maxLoc;
 double min, max;

 // The following line finds the min and max pixel values
 // and their locations in an image.
 minMaxLoc(imLabels, &min, &max, &minLoc, &maxLoc);
 
 // Normalize the image so the min value is 0 and max value is 255.
 imLabels = 255 * (imLabels - min) / (max - min);
 
 // Convert image to 8-bits
 imLabels.convertTo(imLabels, CV_8U);
 
 // Apply a color map
 Mat imColorMap;
 applyColorMap(imLabels, imColorMap, COLORMAP_JET);

 return imColorMap;
}




// Main function
int main() {
  string imagePath = "/home/oa/Dropbox/code/darkest/resources/images/CoinsB.png";
  Mat image = imread(imagePath);
  Mat imageCopy = image.clone();
  // Create a window to display results and set the flag to Autosize
  string windowName = "Show Image";
  namedWindow(windowName, WINDOW_NORMAL);
  ///
  /// YOUR CODE HERE
  ///
  Mat imageGray;
  cvtColor(imageCopy, imageGray, COLOR_BGR2GRAY);
  ///
  /// YOUR CODE HERE
  ///
  Mat channel[3];
  split(imageCopy, channel);
  Mat imageB = channel[0];
  Mat imageG = channel[1];
  Mat imageR = channel[2];
  ///
  /// YOUR CODE HERE
  ///
  Mat thresholdedImage = imageGray.clone();
  threshold(channel[0], thresholdedImage, 128, 255, THRESH_BINARY);
  ///
  /// YOUR CODE HERE
  ///
  Mat kernel1 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
  Mat kernel2 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));
  Mat imageEroded;
  erode(thresholdedImage, imageEroded, kernel1);
  ///
  /// YOUR CODE HERE
  ///
  Mat imageDilated;
  dilate(imageEroded, imageDilated, kernel1);
  dilate(imageDilated, imageDilated, kernel1);
  // Setup SimpleBlobDetector parameters.
  SimpleBlobDetector::Params params;
  params.blobColor = 0;
  params.minDistBetweenBlobs = 2;
  // Filter by Area
  params.filterByArea = false;
  // Filter by Circularity
  params.filterByCircularity = true;
  params.minCircularity = 0.8;
  // Filter by Convexity
  params.filterByConvexity = true;
  params.minConvexity = 0.8;
  // Filter by Inertia
  params.filterByInertia = true;
  params.minInertiaRatio = 0.8;
  // Set up detector with params
  Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
  // Detect blobs
  ///
  /// YOUR CODE HERE
  ///
  // Detect blobs in the image
  vector<KeyPoint> keypoints;
  detector->detect(255-imageDilated, keypoints);
  // Print number of coins detected
  ///
  /// YOUR CODE HERE
  ///
  std::cout << "Number of coins detected: " << keypoints.size() << std::endl;
  // Mark coins using image annotation concepts we have studied so far
  int x, y;
  int radius;
  double diameter;
  ///
  /// YOUR CODE HERE
  ///
  // Draw detected blobs on the image
  for (KeyPoint& keypoint : keypoints) {
    circle(imageCopy, keypoint.pt, keypoint.size, Scalar(0, 255, 0), 1);
    drawMarker(imageCopy, keypoint.pt, Scalar(0, 255, 0), MARKER_CROSS, keypoint.size, 1);
  }
  Mat colorMap = displayConnectedComponents(imageDilated);
  ///
  /// Detect coins using Contour Detection
  ///
  ///
  /// YOUR CODE HERE
  ///
  // Apply Gaussian blur to reduce noise
  Mat blurredImage;
  GaussianBlur(imageDilated, blurredImage, Size(7, 7), 0);

  // Apply Canny edge detection
  Mat edgesImage;
  Canny(imageDilated, edgesImage, 50, 150);

  // Find contours in the binary image
  vector<vector<Point>> contours;
  vector<Vec4i> hierarchy;
  findContours(edgesImage, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  // Draw detected contours on the image
  Mat drawing = image.clone();
  for (vector<Point>& contour : contours) {
    drawContours(drawing, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 2);
  }

  // Remove the inner contours
  // Display the result
  ///
  /// YOUR CODE HERE
  ///
  // Apply Gaussian blur to reduce noise
  Mat blurredImageOuter;
  GaussianBlur(imageDilated, blurredImageOuter, Size(7, 7), 0);

  // Apply Canny edge detection
  Mat edgesImageOuter;
  Canny(blurredImageOuter, edgesImageOuter, 50, 150);

  // Find contours in the binary image
  vector<vector<Point>> contoursOuter;
  vector<Vec4i> hierarchyOuter;
  findContours(edgesImageOuter, contoursOuter, hierarchyOuter, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

  std::cout << "Number of outer contours: " << contoursOuter.size() << std::endl;
  // Draw detected outer contours on the image
  Mat drawingOuter = image.clone();
  for (vector<Point>& contour : contoursOuter) {
    drawContours(drawingOuter, vector<vector<Point>>{contour}, -1, Scalar(0, 0, 255), 2);
  }

  // Print area and perimeter of all contours
  ///
  /// YOUR CODE HERE
  ///
  double area;
  double perimeter;
  for (size_t i=0; i < contours.size(); i++) {
    area = contourArea(contours[i]);
    perimeter = arcLength(contours[i],true);
    cout << "Contour #" << i+1 << " has area = " << area << " and perimeter = " << perimeter << endl;
  }
  // Print maximum area of contour
  // This will be the box that we want to remove
  ///
  /// YOUR CODE HERE
  ///
  double area2;
  double areaMax=0;
  double perimeter2;
  for (size_t i=0; i < contours.size(); i++){
    area2 = contourArea(contours[i]);
    perimeter2 = arcLength(contours[i],true);
    areaMax = (area2 > areaMax) ? area2 : areaMax;
  }
  cout << "Maximum area of contour = " << areaMax << endl;
  // Fit circles on coins
  ///
  /// YOUR CODE HERE
  ///
  Mat fitCirclesImg = image.clone();
  RotatedRect rellipse;
  for (size_t i=0; i < contours.size(); i++){
    // Fit an ellipse
    // We can fit an ellipse only
    // when our contour has minimum
    // 5 points
    if (contours[i].size()<5)
      continue;
    rellipse = fitEllipse(contours[i]);
    ellipse(fitCirclesImg, rellipse, Scalar(255,0,125), 2);
  }

  // Display image
  imshow(windowName, fitCirclesImg);

  waitKey(0);
  destroyAllWindows();
  return 0;
}
