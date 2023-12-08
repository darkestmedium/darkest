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

  // Constructors
  UserData()
    : radius(16)
    , number_of_candidates(16)
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



Mat get_gradient_measures(const vector<Mat>& patches) {

  Mat gradient_scores(patches.size(), 1, CV_32F);

  for (size_t i = 0; i < patches.size(); ++i) {
    Mat gradient_x, gradient_y;

    Sobel(patches[i], gradient_x, CV_32F, 1, 0, 3);
    Sobel(patches[i], gradient_y, CV_32F, 0, 1, 3);

    double mean_gradient_x = mean(abs(gradient_x))[0];
    double mean_gradient_y = mean(abs(gradient_y))[0];

    gradient_scores.at<double>(i) = mean_gradient_x + mean_gradient_y;
  }

  return gradient_scores;
}


void apply_seamless_clone(const Mat& gradient_min_patch, Mat& image, const Point& position) {
  int radius = gradient_min_patch.rows / 2;
  Mat mask = Mat::ones(gradient_min_patch.size(), gradient_min_patch.type()) * 255;
  seamlessClone(gradient_min_patch, image, mask, position, image, NORMAL_CLONE);
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
      // Mat gradients = get_gradient_measures(patches);


      // // Find the index of the patch with the minimum gradient
      // int gradient_min_idx;
      // minMaxIdx(gradients, nullptr, nullptr, &gradient_min_idx);
      
      // cout << patches.size() << endl;
      // cout << gradients << endl;
  
      // // Retrieve the patch with the minimum gradient
      // Mat gradient_min_patch = patches[gradient_min_idx];
    
      // apply_seamless_clone(gradient_min_patch, data->image, position);

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

  UserData data;
  setMouseCallback(args.winName, lmb, &data);


  Mat image = imread(args.filePath);
  if (image.empty()) {cout<<"Can't read file '"<<args.filePath<<"'\n"; return EXIT_FAILURE;}
  data.image = image;
  data.dummy = image;
  data.size = image.size();

  switch(waitKey(1)) {
    case 'c':
      cout << "Key pressed: 'c'" << endl;
      data.image = data.dummy;
      break;
    case 27:
      cout << "Key pressed: 'esc'. Stopping the video" << endl;
      return EXIT_FAILURE;
  }
  imshow(args.winName, image);
  waitKey(0);
  return EXIT_SUCCESS;
}
