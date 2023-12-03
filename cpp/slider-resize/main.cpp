// Import Packages
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// Using namespace to nullify use of cv::function(); syntax
using namespace std;
using namespace cv;
 
int maxScaleUp = 100;
int scaleFactor = 50;

string windowName = "Resize Image";
string trackbarValue = "Scale";


// Callback functions
void scaleImage(int, void*) {
  // Read the image
  Mat image = imread("/home/oa/Dropbox/code/darkest/resources/images/underwater.png");

  // Get the Scale factor from the trackbar
  double scaleFactorDouble = 0.5 + scaleFactor * 0.01;

  Mat scaledImage;
  // Resize the images
  resize(image, scaledImage, Size(), scaleFactorDouble, scaleFactorDouble, INTER_LINEAR);
  // Display the image
  imshow(windowName, scaledImage);
}


int main() {
  // load an image
  Mat image = imread("/home/oa/Dropbox/code/darkest/resources/images/underwater.png");

  // Create a window to display results and set the flag to Autosize
  namedWindow(windowName, WINDOW_AUTOSIZE);

  // Create Trackbars and associate a callback function
  createTrackbar(trackbarValue, windowName, &scaleFactor, maxScaleUp, scaleImage);
  scaleImage(25,0);

  // Display image
  imshow(windowName, image);
  waitKey(0);
  destroyAllWindows();
  return 0;
}