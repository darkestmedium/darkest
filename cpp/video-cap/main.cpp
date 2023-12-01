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




// Variance of absolute values of Laplacian - 
double var_abs_laplacian(Mat image) {
    if (image.channels() > 1) {cvtColor(image, image, COLOR_BGR2GRAY);}
    // Calculate the Laplacian of the image
    Mat laplacian;
    Laplacian(image, laplacian, CV_32F, 3, 1, 0);

    // Calculate the mean of the absolute values of the Laplacian
    double meanAbsLaplacian = 0.0;
    for (int i = 0; i < laplacian.rows; ++i) {
      for (int j = 0; j < laplacian.cols; ++j) {
        meanAbsLaplacian += abs(laplacian.at<float>(i, j));
      }
    }
    meanAbsLaplacian /= (laplacian.rows * laplacian.cols);

    // Calculate the variance of the absolute values of (laplacian - meanAbsLaplacian)
    double varAbsVals = 0.0;
    for (int i = 0; i < laplacian.rows; ++i) {
      for (int j = 0; j < laplacian.cols; ++j) {
        double val = laplacian.at<float>(i, j) - meanAbsLaplacian;
        varAbsVals += pow(val, 2);
      }
    }
    
    return varAbsVals /= (laplacian.rows * laplacian.cols);
}




// Sum Modified Laplacian (SML) - 
double sum_modified_laplacian(Mat image) {
    if (image.channels() > 1) {cvtColor(image, image, COLOR_BGR2GRAY);}
    // Create the kernels for the modified Laplacian filter
    Mat kernelX = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));
    Mat kernelY = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,7));

    // Apply the modified Laplacian filter to the image
    Mat laplacianX;
    filter2D(image, laplacianX, CV_32F, kernelX);
    Mat laplacianY;
    filter2D(image, laplacianY, CV_32F, kernelY);

    // Calculate the absolute value of the filtered images
    Mat absLaplacianX(abs(laplacianX));
    Mat absLaplacianY(abs(laplacianY));

    // Calculate the modified Laplacian
    Mat modifiedLaplacian;https://www.dropbox.com/scl/fi/u382z4mcu45xbej4xz0x3/Lukas-Biernat-Tech-Reel-2023.mp4?rlkey=c9vsmxzasgdoc7gnzao7s0xtt&dl=0
    add(absLaplacianX, absLaplacianY, modifiedLaplacian);

    // Calculate the sum of the modified Laplacian
    double sumModifiedLaplacian = 0.0;
    for (int i = 0; i < modifiedLaplacian.rows; ++i) {
      for (int j = 0; j < modifiedLaplacian.cols; ++j) {
        double val = modifiedLaplacian.at<float>(i, j);
        sumModifiedLaplacian += val;
      }
    }
    return sumModifiedLaplacian;
}





// Main function
int main() {
  string windowName = "Video Capture";
  namedWindow(windowName, WINDOW_NORMAL);

  // Create a VideoCapture object
  string videoCapturePath = "/home/ccpcpp/Dropbox/code/darkest/resources/video/focus-test.mp4";
  VideoCapture cap(videoCapturePath);

  // Read first frame from the video
  Mat frame;
  cap >> frame;

  // Display total number of frames in the video
  cout << "Total number of frames : " << (int)cap.get(CAP_PROP_FRAME_COUNT);

  // Display image
  // imshow(windowName, greyImage);

  // waitKey(0);
  // destroyAllWindows();
  return 0;
}
