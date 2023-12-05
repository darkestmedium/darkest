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




// function which will be called on mouse input
void lmb(int action, int x, int y, int flags, void *userdata) {
  switch (action) {
    case EVENT_LBUTTONDOWN:
      cout<<"LMB pressed at: "<<x<<" x "<<y<<endl;
      break;
    case EVENT_LBUTTONUP:
      cout<<"LMB released at: "<<x<<" x "<<y<<endl;
      break;
  }
};




struct Syntax : public argparse::Args {
  int &width            = kwarg("w,width", "Stream width.").set_default(1280);
  int &height           = kwarg("h,height", "Stream height.").set_default(720);
  int &fps              = kwarg("fps,framerate", "Framerate.").set_default(30);
  int &camera           = kwarg("cam,camera", "Camera input - default is 0.").set_default(0);
  int &mirror           = kwarg("mir,mirror", "Mirror the camera input.").set_default(1);
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "camera";

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
  if(args.verbose) {args.print();}

  namedWindow(args.winName, WINDOW_NORMAL);


  VideoCapture cap(args.camera);
  if(!cap.isOpened()) {cout << "Error opening video stream or file: " << args.camera << endl;}
  cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
  cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);


  setMouseCallback(args.winName, lmb);


  // Read until video is completed
  while(cap.isOpened()) {
    Mat frame;
    cap >> frame;
    // If the frame is empty, break immediately
    if(frame.empty()) {break;}

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
