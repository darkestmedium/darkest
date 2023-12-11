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


struct UserData {
  /* Struct with user data for callback functions.
  */
  Mat image, imgout;
  int trackbar;
  string winName;

  // Constructors
  UserData(string window)
    : winName(window)
  {};
  // Destructors
  ~UserData() {};
};


Mat preProcess(const Mat& image) {
  // Convert to grayscale
  Mat imgrey;
  cvtColor(image, imgrey, COLOR_BGR2GRAY);
  Mat imgblur;
  medianBlur(imgrey, imgblur, 7);
  Mat imgedges;
  Canny(imgblur, imgedges, 0, 50);

  return imgedges;
}


Mat getContours(const Mat& points) {
  // Find contours in the edged image
  vector<vector<Point>> contours;
  findContours(points, contours, RETR_LIST, CHAIN_APPROX_NONE);
  sort(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
    return contourArea(a) > contourArea(b);
  });

  // Get approximate contour
  // Mat target;
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

  return target;
  // return make_pair(uwrap(target), target);
}


Mat uwrap(const Mat& target) {
  Mat hnew = Mat::zeros(4, 2, CV_32S);
  // Yeah, the reduce sum function is not wanna work so we do it manually for now
  Mat add(1, 4, CV_32S);
  add.col(0) = target.at<int>(0,0) + target.at<int>(0,1);
  add.col(1) = target.at<int>(1,0) + target.at<int>(1,1);
  add.col(2) = target.at<int>(2,0) + target.at<int>(2,1);
  add.col(3) = target.at<int>(3,0) + target.at<int>(3,1);

  Mat diff(1, 4, CV_32S);
  diff.col(0) = target.at<int>(0,1) - target.at<int>(0,0);
  diff.col(1) = target.at<int>(1,1) - target.at<int>(1,0);
  diff.col(2) = target.at<int>(2,1) - target.at<int>(2,0);
  diff.col(3) = target.at<int>(3,1) - target.at<int>(3,0);


  Point min_add, max_add, min_diff, max_diff;
  minMaxLoc(add, nullptr, nullptr, &min_add, &max_add);
  minMaxLoc(diff, nullptr, nullptr, &min_diff, &max_diff);

  hnew.at<Point2i>(0) = target.at<Point2i>(min_add.x);
  hnew.at<Point2i>(2) = target.at<Point2i>(max_add.x);
  hnew.at<Point2i>(1) = target.at<Point2i>(min_diff.x);
  hnew.at<Point2i>(3) = target.at<Point2i>(max_diff.x);

  return hnew;
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


Mat unwarp(const Mat& image, Mat approx, Size resolution) {
  Mat persptransform = (cv::Mat_<float>(4, 2) <<
    0, 0,
    resolution.width, 0,
    resolution.width, resolution.height,
    0, resolution.height
  );
  Mat m32f_approx;
  approx.convertTo(m32f_approx, CV_32F);
  Mat M = getPerspectiveTransform(m32f_approx, persptransform);
  Mat imgout;
  warpPerspective(image, imgout, M, resolution);
  imgout.convertTo(imgout, CV_8UC1);
  return imgout;
}


void lmb(int action, int x, int y, int flags, void *userdata) {
  if (userdata == nullptr) {  // Handle null pointer gracefully
    return;
  }
  UserData* data = static_cast<UserData*>(userdata);

  // Mark the top left corner when left mouse button is pressed
  switch(action) {
    case EVENT_LBUTTONUP:
      // Ensure imgout is valid before converting and saving
      if (!data->imgout.empty()) {
        if (data->trackbar>0) {
          detailEnhance(data->imgout, data->imgout, 10, 0.15f);
        }
        imwrite("document.png", data->imgout);
        cout << "Image saved!" << endl;
      } else {
        cerr << "Output image is empty!" << endl;
      }
      break;
  }
}


// Callback functions
void trackbar(int val, void* userdata) {
  UserData* data = static_cast<UserData*>(userdata);
  data->trackbar = val;
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
  UserData data(args.winName);
  setMouseCallback(args.winName, lmb, &data);

  createTrackbar("Trackbar", args.winName, 0, 1, trackbar, &data);

  Mat image = imread(args.filePath);
  if (image.empty()) {
    cout << "Can't read file '" << args.filePath << "'\n";
    return EXIT_FAILURE;
  }
  data.image = image;

  Mat imgedges = preProcess(image);
  Mat target = getContours(imgedges);
  Mat approx = uwrap(target);
  Size resolution = getResolution(target);
  data.imgout = unwarp(image, approx, resolution);


  // drawContours(image, target, -1, Scalar(0, 255, 0), 4);  // draws only four dots :/
  int n = target.rows;
  for(int i = 0 ; i < n ; i++) {
    cv::line(
      image, 
      cv::Point(target.at<int>(i,0), target.at<int>(i,1)), 
      cv::Point(target.at<int>((i+1) % n,0), target.at<int>((i+1) % n,1)), 
      cv::Scalar(0, 255, 0), 2
    );
  }

  imshow(args.winName, data.image);
  waitKey(0);
  return EXIT_SUCCESS;
}
