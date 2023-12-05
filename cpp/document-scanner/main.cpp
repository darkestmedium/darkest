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





cv::Mat unwarp(const cv::Mat& h) {
  cv::Mat hnew(4, 2, CV_32F);

  cv::Mat reshaped_h = h.reshape(2, 4);
  cv::Mat add = reshaped_h.col(0) + reshaped_h.col(1);
  cv::Mat diff = reshaped_h.col(0) - reshaped_h.col(1);

  cv::Point min_add, max_add, min_diff, max_diff;
  cv::minMaxLoc(add, nullptr, nullptr, &min_add, &max_add);
  cv::minMaxLoc(diff, nullptr, nullptr, &min_diff, &max_diff);

  hnew.row(0) = reshaped_h.row(min_add.y);
  hnew.row(2) = reshaped_h.row(max_add.y);
  hnew.row(1) = reshaped_h.row(min_diff.y);
  hnew.row(3) = reshaped_h.row(max_diff.y);

  return hnew;
}


std::pair<cv::Mat,cv::Mat> getContours(const cv::Mat& points) {
  // Find contours in the edged image
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(points, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
  std::sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
    return cv::contourArea(a) > cv::contourArea(b);
  });

  // Get approximate contour
  cv::Mat target;
  for (const auto& contour : contours) {
    double pt = cv::arcLength(contour, true);
    cv::Mat approx;
    cv::approxPolyDP(contour, approx, 0.02 * pt, true);
    if (approx.rows == 4) {
      target = approx;
      break;
    }
  }

  return std::make_pair(unwarp(target), target);
}


cv::Size getResolution(const cv::Mat& contours, const std::string& format = "a4") {
  cv::Point min_point = cv::Point(INT_MAX, INT_MAX);
  cv::Point max_point = cv::Point(INT_MIN, INT_MIN);

  // Find minimum and maximum points in contours
  for (int i = 0; i < contours.rows; ++i) {
    cv::Point point = contours.at<cv::Point>(i, 0);
    min_point.x = std::min(min_point.x, point.x);
    min_point.y = std::min(min_point.y, point.y);
    max_point.x = std::max(max_point.x, point.x);
    max_point.y = std::max(max_point.y, point.y);
  }

  // Calculate width and height
  int width = max_point.x - min_point.x;
  int height = max_point.y - min_point.y;

  std::string mode = "portrait";
  if (width < height) {
    mode = "portrait";
    std::cout << "Mode: Portrait" << std::endl;  // Replace with your logging mechanism
  }
  if (width > height) {
    mode = "landscape";
    std::cout << "Mode: Landscape" << std::endl;  // Replace with your logging mechanism
  }

  int width_mm, height_mm;

  if (format == "a4") {
    if (mode == "portrait") {
      width_mm = 210;
      height_mm = 297;
    } else if (mode == "landscape") {
      width_mm = 297;
      height_mm = 210;
    }
  }

  double aspect_ratio = static_cast<double>(width_mm) / height_mm;
  if (width / height > aspect_ratio) {
    height = static_cast<int>(width / aspect_ratio);
  } else {
    width = static_cast<int>(height * aspect_ratio);
  }

  return cv::Size(width, height);
}






// function which will be called on mouse input
void lmb(int action, int x, int y, int flags, void *userdata) {
  // Mark the top left corner when left mouse button is pressed
  if(action == EVENT_LBUTTONDOWN) {
    cout<<"LMB pressed at: "<<x<<" x "<<y<<endl;
  }
  else if(action == EVENT_LBUTTONUP) {
    cout<<"LMB released at: "<<x<<" x "<<y<<endl;
  }
}


// Callback functions
void trackbar(int val, void*) {
  cout << "Trackbar: " << val << endl;
}




struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/scanned-form.jpg");
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "document-scanner";

  void displayHelp() {
    cout <<
      "Command: "<<commandName<<"\n\n"
      "Flags:\n"
      "  --fp  --filePath (string):  Path to the file, ex. 'image.jpg'.\n"
      "  --wn  --winName (string):  Name of the window.\n"
      "  --v   --verbose (flag):  Toggle Verbose mode.\n\n"
      "Example usage :\n"
      "  "<<commandName<<" --filePath imgage1.jpg\n"
    << endl;
  };
};




int main(int argc, char* argv[]) {
  auto args = argparse::parse<Syntax>(argc, argv);

  if(args.help) {args.displayHelp(); return EXIT_FAILURE;}
  if(args.verbose) args.print();

  namedWindow(args.winName, WINDOW_NORMAL);


  setMouseCallback(args.winName, lmb);
  createTrackbar("Trackbar", args.winName, 0, 100, trackbar);
  trackbar(25, 0);



  Mat image = imread(args.filePath);
  if (image.empty()) {
    cout << "Can't read file '" << args.filePath << "'\n";
    return EXIT_FAILURE;
  }

  // Convert to grayscale
  cv::Mat imgrey;
  cv::cvtColor(image, imgrey, cv::COLOR_BGR2GRAY);
  cv::Mat imgblur;
  cv::medianBlur(imgrey, imgblur, 7);
  cv::Mat imgedges;
  cv::Canny(imgblur, imgedges, 0, 50);


  // Assume you have implemented get_contours and get_resolution functions
  auto [approx, target] = getContours(image);

  cv::Size resolution = getResolution(approx);

  // Get the perspective transform
  cv::Mat persptrans = cv::getPerspectiveTransform(approx, cv::Mat_<float>(4, 2) << 0, 0, width, 0, width, height, 0, height);

  // Warp the perspective
  cv::Mat imgout;
  cv::warpPerspective(image, imgout, persptrans, resolution);

  // Draw contours on the original image
  std::vector<std::vector<cv::Point>> contours;
  contours.push_back(target);
  cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), 2);






  imshow(args.winName, imgedges);
  waitKey(0);
  return EXIT_SUCCESS;
}
