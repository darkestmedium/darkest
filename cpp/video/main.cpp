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
  if(!cap.isOpened()) {cout << "Error opening video stream or file: " << args.filePath << endl;}

  // Display total number of frames in the video
  // cout << "Total number of frames : " << int(cap.get(CAP_PROP_FRAME_COUNT)) << endl;

  // Read until video is completed
  while(cap.isOpened()) {
    Mat frame;
    cap >> frame;
    // If the frame is empty, break immediately
    if(frame.empty()) break;

    switch(waitKey(args.fps)) {
      case 'c':
        cout << "Key pressed: 'c'" << endl;
      case 27: // esc is pressed
        cout << "Key pressed: 'esc'. Stopping the video" << endl;
        return EXIT_FAILURE;
      // default: // -1 is returned and printed on every frame
      //   cout << "Key pressed: " << key << endl;
    }
    imshow(args.winName, frame);
  }

  return EXIT_SUCCESS;
}
