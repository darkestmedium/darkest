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



void printHelp() {
  cout <<
    "image command.\n\n"
    "Flags:\n"
    "  --fp  --filePath (string):  Path to the file, ex. 'image.jpg'.\n"
    "  --wn  --winName (string):  Name of the window.\n"
    "  --v   --verbose (flag):  Toggle Verbose mode.\n"
    "Example usage :\n"
    "  image --filePath imgage1.jpg\n";
};


struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/blemish.png");
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");
};



int main(int argc, char* argv[]) {
  auto args = argparse::parse<Syntax>(argc, argv);

  if (args.help) {printHelp(); return EXIT_FAILURE;}
  if (args.verbose) args.print();

  namedWindow(args.winName, WINDOW_NORMAL);

  Mat image = imread(args.filePath);


  Mat imageCopy = image.clone();
  // Mat greyImage = rgbToGray(imageCopy);
  // Mat hsvImage = convertBGRtoHSV(imageCopy);


  imshow(args.winName, image);
  waitKey(0);

  return EXIT_SUCCESS;
}
