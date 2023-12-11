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




Mat unwarp(const Mat& h) {
  Mat hnew(2, 4, CV_64F);

  Mat reshaped_h = h.reshape(2, 4);
  Mat add = reshaped_h.col(0) + reshaped_h.col(1);
  Mat diff = reshaped_h.col(0) - reshaped_h.col(1);

  Point min_add, max_add, min_diff, max_diff;
  minMaxLoc(add, nullptr, nullptr, &min_add, &max_add);
  minMaxLoc(diff, nullptr, nullptr, &min_diff, &max_diff);

  hnew.row(0) = reshaped_h.row(min_add.y);
  hnew.row(2) = reshaped_h.row(max_add.y);
  hnew.row(1) = reshaped_h.row(min_diff.y);
  hnew.row(3) = reshaped_h.row(max_diff.y);

  return hnew;
}


pair<Mat, Mat> getContours(const Mat& points) {
  // Find contours in the edged image
  vector<vector<Point>> contours;
  findContours(points, contours, RETR_LIST, CHAIN_APPROX_NONE);
  sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
    return contourArea(a) > contourArea(b);
  });

  // Get approximate contour
  Mat target;
  // Mat target = Mat(Size(2, 4), CV_64F);
  for (const auto& contour : contours) {
    float pt = arcLength(contour, true);
    Mat approx;
    approxPolyDP(contour, approx, 0.02 * pt, true);
    if (approx.rows == 4) {
      target = approx;
      break;
    }
  }

  return make_pair(unwarp(target), target);
}


Size getResolution(const Mat& contours, const string& format = "a4") {
  Point min_point = Point(INT_MAX, INT_MAX);
  Point max_point = Point(INT_MIN, INT_MIN);

  // Find minimum and maximum points in contours
  for (int i = 0; i < contours.rows; ++i) {
    Point point = contours.at<Point>(i, 0);
    min_point.x = min(min_point.x, point.x);
    min_point.y = min(min_point.y, point.y);
    max_point.x = max(max_point.x, point.x);
    max_point.y = max(max_point.y, point.y);
  }

  // Calculate width and height
  int width = max_point.x - min_point.x;
  int height = max_point.y - min_point.y;

  string mode = "portrait";
  if (width < height) {
    mode = "portrait";
    cout << "Mode: Portrait" << endl;  // Replace with your logging mechanism
  }
  if (width > height) {
    mode = "landscape";
    cout << "Mode: Landscape" << endl;  // Replace with your logging mechanism
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

  float aspect_ratio = static_cast<float>(width_mm) / height_mm;
  if (width / height > aspect_ratio) {
    height = static_cast<int>(width / aspect_ratio);
  } else {
    width = static_cast<int>(height * aspect_ratio);
  }

  return Size(width, height);
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
  string &filePath      = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/scanned-form.jpg");
  string &winName       = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
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

  Mat image = imread(args.filePath);
  if (image.empty()) {
    cout << "Can't read file '" << args.filePath << "'\n";
    return EXIT_FAILURE;
  }

  // Convert to grayscale
  Mat imgrey;
  cvtColor(image, imgrey, COLOR_BGR2GRAY);
  Mat imgblur;
  medianBlur(imgrey, imgblur, 7);
  Mat imgedges;
  Canny(imgblur, imgedges, 0, 50);


  // Find contours in the edged image
  vector<vector<Point>> contours;
  findContours(imgedges, contours, RETR_LIST, CHAIN_APPROX_NONE);
  sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
    return contourArea(a) > contourArea(b);
  });

  // Get approximate contour
  Mat target;
  for (const auto& contour : contours) {
    float pt = arcLength(contour, true);
    Mat approx;
    approxPolyDP(contour, approx, 0.02 * pt, true);
    if (approx.rows == 4) {
      target = approx;
      break;
    }
  }

  // cout << target << endl;

  Mat hnew = Mat::zeros(4, 2, CV_32F);
  Mat reshaped_h = target.reshape(0, 4);

  // Yeah, the reduce sum function is not wanna work so we do it manually for now
  Mat add(1, 4, CV_32F);
  add.col(0) = target.at<int>(0,0) + target.at<int>(0,1);
  add.col(1) = target.at<int>(1,0) + target.at<int>(1,1);
  add.col(2) = target.at<int>(2,0) + target.at<int>(2,1);
  add.col(3) = target.at<int>(3,0) + target.at<int>(3,1);

  Mat diff(1, 4, CV_32F);
  diff.col(0) = target.at<int>(0,1) - target.at<int>(0,0);
  diff.col(1) = target.at<int>(1,1) - target.at<int>(1,0);
  diff.col(2) = target.at<int>(2,1) - target.at<int>(2,0);
  diff.col(3) = target.at<int>(3,1) - target.at<int>(3,0);


  // cout << add << endl;
  // cout << diff << endl;

  Point min_add, max_add, min_diff, max_diff;
  minMaxLoc(add, nullptr, nullptr, &min_add, &max_add);
  minMaxLoc(diff, nullptr, nullptr, &min_diff, &max_diff);

  // cout << target << endl;

  hnew.row(0) = reshaped_h.row(min_add.y);
  hnew.row(2) = reshaped_h.row(max_add.y);
  hnew.row(1) = reshaped_h.row(min_diff.y);
  hnew.row(3) = reshaped_h.row(max_diff.y);

  hnew.row(0) = target.row(min_add.x);
  hnew.row(2) = reshaped_h.row(max_add.y);
  hnew.row(1) = reshaped_h.row(min_diff.y);
  hnew.row(3) = reshaped_h.row(max_diff.y);

  // Mat uw_target = unwarp(target);
  // cout << hnew << endl;

  // return make_pair(unwarp(target), target);


  Size resolution = getResolution(target);
  // cout << resolution.width << endl;
  // cout << resolution.height << endl;

  // Get the perspective transform
  Mat persptransform = (cv::Mat_<float>(4, 2) <<
    0, 0,
    resolution.width, 0,
    resolution.width, resolution.height,
    0, resolution.height
  );
  Mat targetWarp;
  target.convertTo(targetWarp, CV_32F);
  Mat M = getPerspectiveTransform(targetWarp, persptransform);
  Mat imgout;
  warpPerspective(image, imgout, M, resolution);


  drawContours(image, target, -1, Scalar(0, 255, 0), 4);


  imshow(args.winName, image);
  waitKey(0);
  return EXIT_SUCCESS;
}
