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



cv::Mat get_outline(const cv::Mat& image, int blur=5, int edge=9) {
  cv::Mat imageBlur;
  cv::medianBlur(image, imageBlur, blur);

  cv::Mat result;
  cv::adaptiveThreshold(imageBlur, result, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, edge, blur);

  return result;
}


cv::Mat color_quantize(const cv::Mat& image, int colors=8) {
  cv::Mat data = image;
  data.convertTo(data, CV_32F);

  cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::MAX_ITER, 20, 0.001);

  cv::Mat center, label;
  // auto label;
  cv::kmeans(data, colors, label, criteria, 10, cv::KMEANS_RANDOM_CENTERS, center);
  center.convertTo(center, CV_8U);

  return center;
}



cv::Mat pencilSketch(const cv::Mat& image, int arguments=0, int blur=5, int edge=9, double alpha=0.5) {
  cv::Mat imgrey;
  cv::cvtColor(image, imgrey, cv::COLOR_BGR2GRAY);

  cv::Mat imgreyquant = color_quantize(imgrey);
  cv::Mat imageblur;
  cv::medianBlur(imgreyquant, imageblur, blur);

  cv::Mat imoutline;
  cv::Mat imoutlinecol;
  cv::pencilSketch(image, imoutline, imoutlinecol, 60.0f, 0.075f, 0.085f);

  cv::Mat pencilSketchImage;
  cv::addWeighted(imageblur, 1 - alpha, imoutline, alpha, 0, pencilSketchImage);

  cv::cvtColor(pencilSketchImage, pencilSketchImage, cv::COLOR_GRAY2BGR);

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
  // Mat imageCopy = image.clone();


  cv::Mat imgrey;
  cv::cvtColor(image, imgrey, cv::COLOR_BGR2GRAY);
  Mat imoutline = get_outline(imgrey);


  imshow(args.winName, imoutline);
  // imshow(args.winName, image);

  waitKey(0);
  return EXIT_SUCCESS;
}
