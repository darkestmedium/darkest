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



Mat convertBGRtoGray(Mat image) {
  Mat greyImage(image.size(), CV_8UC1);
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      // Convert each pixel to grayscale using the given formula
      uchar blue = image.at<cv::Vec3b>(i, j)[0];
      uchar green = image.at<cv::Vec3b>(i, j)[1];
      uchar red = image.at<cv::Vec3b>(i, j)[2];
      greyImage.at<uchar>(i, j) = 0.114 * blue  + 0.587 * green +  0.299 * red;
    }
  }
  return greyImage;
}




Mat convertBGRtoHSV(Mat image) {
Mat hsvImage(image.size(), CV_8UC3);

for (int i = 0; i < image.rows; ++i) {
  for (int j = 0; j < image.cols; ++j) {
    // Convert BGR to HSV using the OpenCV formula
    uchar blue = image.at<cv::Vec3b>(i, j)[0];
    uchar green = image.at<cv::Vec3b>(i, j)[1];
    uchar red = image.at<cv::Vec3b>(i, j)[2];

    float b = blue / 255.0f;
    float g = green / 255.0f;
    float r = red / 255.0f;

    float cmax = std::max({r, g, b});
    float cmin = std::min({r, g, b});
    float delta = cmax - cmin;

    float hue = 0;

    // Calculate hue
    if (delta != 0) {
      if (cmax == r) {
        hue = 60 * fmod(((g - b) / delta), 6.0);
      } else if (cmax == g) {
        hue = 60 * (((b - r) / delta) + 2);
      } else if (cmax == b) {
        hue = 60 * (((r - g) / delta) + 4);
      }
    }

    if (hue < 0) {
      hue += 360;
    }

    // Calculate saturation
    float saturation = (cmax != 0) ? (delta / cmax) : 0;
    // Calculate value
    float value = cmax;
    // Set HSV values in the output image
    hsvImage.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(hue / 2);
    hsvImage.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(saturation * 255);
    hsvImage.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(value * 255);
  }
}

return hsvImage;
}




// Main function
int main() {

  string windowName = "Show Image";
  namedWindow(windowName, WINDOW_NORMAL);

  string imagePath = "/home/oa/Dropbox/code/darkest/resources/images/CoinsA.png";
  Mat image = imread(imagePath);
  Mat imageCopy = image.clone();

  Mat greyImage = rgbToGrey(imageCopy);
  Mat hsvImage = convertBGRtoHSV(imageCopy);

  // Display image
  imshow(windowName, greyImage);
  waitKey(0);
  destroyAllWindows();
  return 0;
}
