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




// Main function
int main() {
  string windowName = "Show Image";
  namedWindow(windowName, WINDOW_NORMAL);

  string imagePath = "/home/oa/Dropbox/code/darkest/resources/images/CoinsA.png";
  Mat image = imread(imagePath);
  Mat imageCopy = image.clone();

  Mat greyImage = rgbToGray(imageCopy);
  Mat hsvImage = convertBGRtoHSV(imageCopy);

  // Display image
  imshow(windowName, greyImage);
  waitKey(0);
  destroyAllWindows();
  return 0;
}
