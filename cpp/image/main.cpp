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




void lmb(int action, int x, int y, int flags, void *userdata) {
  // Mark the top left corner when left mouse button is pressed
  switch (action) {
    case EVENT_LBUTTONDOWN:
      cout<<"LMB pressed at: "<<x<<" x "<<y<<endl;
      break;
    case EVENT_LBUTTONUP:
      cout<<"LMB released at: "<<x<<" x "<<y<<endl;
      break;
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

  setMouseCallback(args.winName, lmb);

  Mat image = imread(args.filePath);
  if (image.empty())  {
    cout << "Can't read file '" << args.filePath << "'\n";
    return EXIT_FAILURE;
  }
  // Mat imageCopy = image.clone();

  imshow(args.winName, image);
  waitKey(0);
  return EXIT_SUCCESS;
}
