// System includes
#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"

// Custom includes
#include "../api/argparse.hpp"




using namespace std;
using namespace cv;


// // Implement Variance of absolute values of Laplacian 
// double var_abs_laplacian(Mat image) {
//   ///
//   /// YOUR CODE HERE
//   ///
//   if (image.channels() > 1) {cvtColor(image, image, COLOR_BGR2GRAY);}

//   Mat laplacian;
//   Laplacian(image, laplacian, CV_32F, 3, 1, 0);

//   // Calculate the mean of the absolute values of the Laplacian
//   double meanAbsLaplacian = 0.0;
//   for (int i = 0; i < laplacian.rows; ++i) {
//     for (int j = 0; j < laplacian.cols; ++j) {
//       meanAbsLaplacian += abs(laplacian.at<float>(i, j));
//     }
//   }
//   meanAbsLaplacian /= (laplacian.rows * laplacian.cols);

//   // Calculate the variance of the absolute values of (laplacian - meanAbsLaplacian)
//   double varAbsVals = 0.0;
//   for (int i = 0; i < laplacian.rows; ++i) {
//     for (int j = 0; j < laplacian.cols; ++j) {
//       double val = laplacian.at<float>(i, j) - meanAbsLaplacian;
//       varAbsVals += pow(val, 2);
//     }
//   }
//   return varAbsVals /= (laplacian.rows * laplacian.cols);
// }


double var_abs_laplacian(Mat image) {
  Mat laplacian = Mat(image.size(), CV_32F);

  // Calculate the Laplacian of the image
  Laplacian(image, laplacian, CV_32F, 3, 1, 0);

  // Calculate the mean of the absolute values of the Laplacian
  Mat absolute_laplacian = abs(laplacian);
  double mean_absolute_laplacian = mean(absolute_laplacian)[0];

  // Calculate the variance of the absolute values of the Laplacian
  Mat centered_absolute_laplacian = absolute_laplacian - mean_absolute_laplacian;
  Mat variance_absolute_laplacian = centered_absolute_laplacian.mul(centered_absolute_laplacian);
  double sum_variance_absolute_laplacian = sum(variance_absolute_laplacian)[0];
  double variance = sum_variance_absolute_laplacian / (image.rows * image.cols - 1);

  // Return the variance of the absolute values of the Laplacian
  return variance;
}


double sum_modified_laplacian(Mat image) {
  // Check if the input image is empty
  // if (image.empty()) {
  //   return 0.0;
  // }

  // // Convert the input image to grayscale if it's not already grayscale
  // if (image.channels() != 1) {
  //   Mat grayImage;
  //   cvtColor(image, grayImage, COLOR_BGR2GRAY);
  //   return sum_modified_laplacian(grayImage);
  // }

  // Calculate the modified Laplacian
  Mat laplacian_x = Mat(image.size(), CV_32F);
  Mat laplacian_y = Mat(image.size(), CV_32F);

  // Define the modified Laplacian filter masks
  Mat kernelX = Mat::zeros(3, 3, CV_8S);
  kernelX.at<int>(1, 1) = 2;
  kernelX.at<int>(1, 0) = -1;
  kernelX.at<int>(1, 2) = -1;

  Mat kernelY = Mat::zeros(3, 3, CV_8S);
  kernelY.at<int>(1, 1) = 2;
  kernelY.at<int>(0, 1) = -1;
  kernelY.at<int>(2, 1) = -1;

  // Apply the modified Laplacian filters
  filter2D(image, laplacian_x, -1, kernelX, Point(-1, -1), 0, BORDER_DEFAULT);
  filter2D(image, laplacian_y, -1, kernelY, Point(-1, -1), 0, BORDER_DEFAULT);

  // Calculate the sum of the absolute values of the modified Laplacian
  Mat absolute_laplacian_x = abs(laplacian_x);
  Mat absolute_laplacian_y = abs(laplacian_y);
  Mat sum_laplacian = absolute_laplacian_x + absolute_laplacian_y;

  // Return the sum of the absolute values of the modified Laplacian
  return sum(sum_laplacian)[0];
}


// // Implement Sum Modified Laplacian
// double sum_modified_laplacian(Mat image) {
//   ///
//   /// YOUR CODE HERE
//   ///
//   if (image.channels() > 1) {cvtColor(image, image, COLOR_BGR2GRAY);}
//   // Create the kernels for the modified Laplacian filter
//   // Define the Laplacian kernels
//   Mat kernelX = Mat::zeros(3, 3, CV_8S);
//   kernelX.at<int>(1, 1) = 2;
//   kernelX.at<int>(1, 0) = -1;
//   kernelX.at<int>(1, 2) = -1;

//   Mat kernelY = Mat::zeros(3, 3, CV_8S);
//   kernelY.at<int>(1, 1) = 2;
//   kernelY.at<int>(0, 1) = -1;
//   kernelY.at<int>(2, 1) = -1;

//   // Apply the modified Laplacian filter to the image
//   Mat laplacianX;
//   filter2D(image, laplacianX, CV_32F, kernelX, Point(-1, -1), 0, BORDER_DEFAULT);
//   Mat laplacianY;
//   filter2D(image, laplacianY, CV_32F, kernelY, Point(-1, -1), 0, BORDER_DEFAULT);

//   // Calculate the absolute value of the filtered images

//   // Calculate the modified Laplacian
//   Mat modifiedLaplacian(abs(laplacianX) + abs(laplacianY));


//   // Calculate the sum of the modified Laplacian
//   double sumModifiedLaplacian = 0.0;
//   for (int i = 0; i < modifiedLaplacian.rows; ++i) {
//     for (int j = 0; j < modifiedLaplacian.cols; ++j) {
//       double val = modifiedLaplacian.at<float>(i, j);
//       sumModifiedLaplacian += val;
//     }
//   }
//   return sumModifiedLaplacian;
// };






struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/video/focus-test.mp4");
  int &fps              = kwarg("fps,framerate", "Playback framerate in ms - 30fps / 33.33ms.").set_default(33);
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "video";

  void displayHelp() {
    cout <<
      "Command: "<<commandName<<"\n\n"
      "Flags:\n"
      "  -fp   --filePath (string):  Path to the file, ex. 'video1.mp4'.\n"
      "  -fps  --framerate (string):  Playback framerate in ms - 30fps / 33.33ms.\n"
      "  -wn   --winName (string):  Name of the opencv window.\n"
      "  -v    --verbose (flag):  Toggle Verbose mode.\n\n"
      "Example usage :\n"
      "  "<<commandName<<" --filePath imgage1.jpg\n"
    << endl;
  }
};




int main(int argc, char* argv[]) {
  auto args = argparse::parse<Syntax>(argc, argv);

  if(args.help) {args.displayHelp(); return EXIT_FAILURE;}
  if(args.verbose) args.print();

  namedWindow(args.winName, WINDOW_NORMAL);


  VideoCapture cap(args.filePath);
  if(!cap.isOpened()) {cout << "Error opening video stream or file: " << args.filePath << endl; return EXIT_FAILURE;}


  // Read first frame from the video
  Mat frame;
  cap >> frame;

  // Display total number of frames in the video
  cout << "Total number of frames : " << int(cap.get(CAP_PROP_FRAME_COUNT)) << endl;

  double maxV1 = 0;
  double maxV2 = 0;


  // Frame with maximum measure of focus
  // Obtained using methods 1 and 2
  Mat bestFrame1;
  Mat bestFrame2;

  // Frame ID of frame with maximum measure
  // of focus
  // Obtained using methods 1 and 2
  int bestFrameId1 = 0;
  int bestFrameId2 = 0;

  // // Get measures of focus from both methods
  double val1 = var_abs_laplacian(frame);
  double val2 = sum_modified_laplacian(frame);

  // Specify the ROI for flower in the frame
  // UPDATE THE VALUES BELOW
  int topCorner = 0;
  int leftCorner = 0;
  int bottomCorner = frame.size().height;
  int rightCorner = frame.size().width;

  Mat flower;
  flower = frame(Range(topCorner, bottomCorner), Range(leftCorner, rightCorner));

  // Iterate over all the frames present in the video
  while (1) {
    // Crop the flower region out of the frame
    flower = frame(Range(topCorner,bottomCorner), Range(leftCorner, rightCorner));
    // Get measures of focus from both methods
    val1 = var_abs_laplacian(flower);
    val2 = sum_modified_laplacian(flower);
    // If the current measure of focus is greater 
    // than the current maximum
    if (val1 > maxV1) {
      // Revise the current maximum
      maxV1 = val1;
      // Get frame ID of the new best frame
      bestFrameId1 = int(cap.get(CAP_PROP_POS_FRAMES));
      // Revise the new best frame
      bestFrame1 = frame.clone();
      cout << "Frame ID of the best frame [Method 1]: " << bestFrameId1 << endl;
    }
    // If the current measure of focus is greater 
    // than the current maximum
    if (val2 > maxV2){
      // Revise the current maximum
      maxV2 = val2;
      // Get frame ID of the new best frame
      bestFrameId2 = int(cap.get(CAP_PROP_POS_FRAMES));
      // Revise the new best frame
      bestFrame2 = frame.clone();
      cout << "Frame ID of the best frame [Method 2]: " << bestFrameId2 << endl;
    }
    cap >> frame;
    if (frame.empty())
      break;
  }

  cout << "================================================" << endl;

  // Print the Frame ID of the best frame
  cout << "Frame ID of the best frame [Method 1]: " << bestFrameId1 << endl;
  cout << "Frame ID of the best frame [Method 2]: " << bestFrameId2 << endl;

  cap.release();

  Mat out;
  hconcat(bestFrame1, bestFrame2, out);

  return EXIT_SUCCESS;
}
