// System includes
#include <iostream>
#include <vector>
#include <cmath>

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
  Mat image, dummy;
  Size size;
  int radius;
  int number_of_candidates;
  string winName;

  // Constructors
  UserData(string window)
    : radius(16)
    , number_of_candidates(16)
    , winName(window)
  {};
  // Destructors
  ~UserData() {};
};




vector<Mat> get_patches(Point position, int& number_of_candidates, int& radius, const Mat& image) {
  // Finding specific number of evenly spaced points around our patch
  Mat t = Mat::zeros(1, number_of_candidates + 1, CV_32F);
  for (int i = 0; i <= number_of_candidates; ++i) {
    t.at<float>(i) = 2 * CV_PI * i / number_of_candidates;
  }

  Mat tCos, tSin;
  t.convertTo(tCos, CV_32F);
  t.convertTo(tSin, CV_32F);
  tCos.forEach<float>([&](float& pixel, const int*) -> void {
    pixel = round(2 * radius * cos(pixel) + position.x);
  });
  tSin.forEach<float>([&](float& pixel, const int*) -> void {
    pixel = round(2 * radius * sin(pixel) + position.y);
  });

  Mat candidate_centers(tCos.cols, 2, CV_32F);
  for (int idx = 0; idx < tCos.cols; ++idx) {
    candidate_centers.at<float>(idx, 0) = tCos.at<float>(idx);
    candidate_centers.at<float>(idx, 1) = tSin.at<float>(idx);
  }

  candidate_centers = candidate_centers.rowRange(Range(0, candidate_centers.rows - 1));

  vector<Mat> patches;
  for (int idx = 0; idx < candidate_centers.rows; ++idx) {
    Point center(candidate_centers.at<float>(idx, 0), candidate_centers.at<float>(idx, 1));
    if (
      0 + radius < center.x
      and center.x < image.cols - radius
      and 0 + radius < center.y 
      and center.y < image.rows - radius
    ) {
      Mat patch = image(Rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius)).clone();
      patches.push_back(patch);
    }
  }

  return patches;
}




double calculateMeanGradient(const cv::Mat& imagePatch, int xOrder, int yOrder) {
  cv::Mat sobel;
  cv::Sobel(imagePatch, sobel, CV_32F, xOrder, yOrder, 3);
  cv::Mat absSobel = cv::abs(sobel);
  double meanGradient = cv::mean(absSobel)[0];
  return meanGradient;
}




cv::Mat calculateCandidateGradientMeasures(const std::vector<cv::Mat>& patches) {
  std::vector<double> sobelX, sobelY;
  // Calculate horizontal (x) and vertical (y) Sobel gradients for each patch
  for (const auto& patch : patches) {
    sobelX.push_back(calculateMeanGradient(patch, 1, 0));
    sobelY.push_back(calculateMeanGradient(patch, 0, 1));
  }
  // Sum the horizontal and vertical gradients
  std::vector<float> totalGradients(patches.size());
  for (size_t i = 0; i < patches.size(); ++i) {
    totalGradients[i] = sobelX[i] + sobelY[i];
  }
  // Convert the vector to a one-dimensional cv::Mat
  cv::Mat resultMat(totalGradients, false);

  return resultMat;
}



void lmb(int action, int x, int y, int flags, void *userdata) {
  // Mark the top left corner when left mouse button is pressed
  UserData* data = static_cast<UserData*>(userdata);

  switch (action) {
    case EVENT_LBUTTONDOWN:
      cout<<"LMB pressed at: "<<x<<" x "<<y<<endl;
      Point position(x, y);

      bool patchFits = (
        0 + data->radius < x
        and x < data->size.width - data->radius)
        and (0 + data->radius < y
        and y < data->size.height - data->radius
      );
      if(!patchFits) {break;}

      vector<Mat> patches = get_patches(position, data->number_of_candidates, data->radius, data->image);
      Mat gradients = calculateCandidateGradientMeasures(patches);

      // Find the index of the patch with the minimum gradient
      int gradient_min_idx;
      minMaxIdx(gradients, nullptr, nullptr, &gradient_min_idx);
      Mat gradient_min_patch = patches[gradient_min_idx];

      gradient_min_patch.convertTo(gradient_min_patch, CV_8UC3);
      data->image.convertTo(data->image, CV_8UC3);

      Mat mask = Mat::ones(gradient_min_patch.size(), CV_8U)*255;

      Mat input(data->image);

      Mat output;

      seamlessClone(gradient_min_patch, data->image, mask, position, output, NORMAL_CLONE);
      data->image = output;
      imshow(data->winName, data->image);
      break;
    // case EVENT_LBUTTONUP:
    //   cout<<"LMB released at: "<<x<<" x "<<y<<endl;
    //   break;
  }
}




struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/blemish.png");
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "camera";

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
  if(args.verbose) {args.print();}

  namedWindow(args.winName, WINDOW_NORMAL);

  UserData data(args.winName);
  setMouseCallback(args.winName, lmb, &data);


  Mat image = imread(args.filePath);
  if (image.empty()) {cout<<"Can't read file '"<<args.filePath<<"'\n"; return EXIT_FAILURE;}
  data.image = image;
  data.dummy = image;
  data.size = image.size();

  int k=0;
  while(k!=27) {  // loop until esc is pressed
    imshow(args.winName, data.image);
    k = waitKey(1);
    if(k == 99) {  // If c is pressed, clear the window, using the dummy image
      cout << "'c' was pressed, image reset." << endl;
      data.image = data.dummy;
      imshow(args.winName, data.image);
    }
  }

  // switch(waitKey(1)) {
  //   case 'c':
  //     cout << "Key pressed: 'c'" << endl;
  //     // data.image = data.dummy;
  //     imshow(args.winName, data.dummy);
  //     break;
  //   case 27:
  //     cout << "Key pressed: 'esc'. Stopping the video" << endl;
  //     return EXIT_FAILURE;
  //   // default: // -1 is returned and printed on every frame
  //   //   imshow(args.winName, data.image);
  //   //   // cout << "Key pressed: " << key << endl;
  //   //   break;
  // }
  
  return EXIT_SUCCESS;
}
