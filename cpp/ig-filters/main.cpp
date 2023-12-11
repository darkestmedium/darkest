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




Mat get_outline(Mat image, int blur=5, int edge=9) {
  Mat imageBlur;
  medianBlur(image, imageBlur, blur);
  Mat result;
  adaptiveThreshold(imageBlur, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, edge, blur);
  return result;
}




Mat cartoonify(Mat image) {
  Mat cartoonImage;

  /// YOUR CODE HERE

  Mat imageblur;
  edgePreservingFilter(image, imageblur, RECURS_FILTER);

  Mat imgoutline = image;
  cvtColor(imgoutline, imgoutline, COLOR_BGR2GRAY);
  adaptiveThreshold(imgoutline, imgoutline, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 9);

  Mat mask = imgoutline;
  cvtColor(imgoutline, imgoutline, COLOR_GRAY2BGR);

  bitwise_and(imageblur, imgoutline, cartoonImage, mask);


  return cartoonImage;
}




Mat color_quantize(const Mat& image, int colors=8) {
  Mat data = image;
  data.convertTo(data, CV_32F);

  TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 0.001);

  Mat center, label;
  kmeans(data, colors, label, criteria, 10, KMEANS_RANDOM_CENTERS, center);
  center.convertTo(center, CV_8U);

  return center;
}




Mat pencilSketch(Mat image) {

  Mat pencilSketchImage;

  Mat imgrey;
  cvtColor(image, imgrey, COLOR_BGR2GRAY);
  Mat imageblur;
  edgePreservingFilter(imgrey, imageblur, RECURS_FILTER);

  Mat imgoutline, imgoutlinecol;
  adaptiveThreshold(imgrey, imgoutline, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 9);
  // pencilSketch(image, imgoutline, imgoutlinecol, 50.0f, 0.1f, 0.1f);

  bitwise_and(imageblur, imgoutline, pencilSketchImage, imgoutline);

  return pencilSketchImage;
}





struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/trump.jpg");
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
  if(args.verbose) args.print();

  namedWindow(args.winName, WINDOW_NORMAL);

  Mat image = imread(args.filePath);
  if (image.empty()) {
    cout << "Can't read file '" << args.filePath << "'\n";
    return EXIT_FAILURE;
  }

  // Mat imgrey = pencilSketch(image);
  Mat imgcartoon = cartoonify(image);


  imshow(args.winName, imgcartoon);

  waitKey(0);
  return EXIT_SUCCESS;
}
