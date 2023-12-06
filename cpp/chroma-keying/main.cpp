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
  Mat image;
  Vec<uint,3> colop;
  Vec<uint,3> color;
  int softness;

  // Constructors
  UserData(Mat image)
    : image(image)
    , colop(68, 255, 0)
    , color(17, 255, 0)
    , softness(0)
  {};
  UserData()
    : colop(68, 255, 0)
    , color(17, 255, 0)
    , softness(0)
  {};
  // Destructors
  ~UserData() {};
};




template <typename T>
inline float get_luminance(Vec<T,3> &color) {
  /* Calculates the color luminance.
  */
  return color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299;
}




void lmb(int action, int x, int y, int flags, void *userdata) {
  /* Mouse callback function.
  */
  UserData* data = static_cast<UserData*>(userdata);

  switch (action) {
    case EVENT_LBUTTONDOWN:
      data->colop = data->image.at<Vec3b>(y, x);
      cout<<"Color sampeled on press: "<<data->colop<<endl;
      break;
    case EVENT_LBUTTONUP:
      data->color = data->image.at<Vec3b>(y, x);
      cout<<"Color sampled on release: "<<data->color<<endl;
      break;
  }

  // Sort colors based on luminance
  float lumiop = get_luminance(data->colop);
  float lumior = get_luminance(data->color);
  if (lumiop > lumior) {
    data->colop = data->color;
    data->color = data->colop;
  }
}



void softness(int trkbVal, void *userdata) {
  UserData* data = static_cast<UserData*>(userdata);
  data->softness = trkbVal;
}



struct Syntax : public argparse::Args {
  /* Syntax struct with args and help printing for the command.
  */
  string &filePath      = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/that-space.png");
  int &width            = kwarg("w,width", "Stream width.").set_default(1280);
  int &height           = kwarg("h,height", "Stream height.").set_default(720);
  int &fps              = kwarg("fps,framerate", "Framerate.").set_default(30);
  int &camera           = kwarg("cam,camera", "Camera input - default is 0.").set_default(0);
  int &mirror           = kwarg("mir,mirror", "Mirror the camera input.").set_default(1);
  string &winName       = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "chroma-keying";

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


  UserData data;
  setMouseCallback(args.winName, lmb, &data);
  createTrackbar("Softness", args.winName, 0, 100, softness, &data);

  

  // Read until video is completed
  while(cap.isOpened()) {
    Mat image;
    cap >> image;
    if(image.empty()) break;

    Mat imgback = imread(args.filePath);
    resize(imgback, imgback, Size(args.width, args.height));

    data.image = image;
    Mat mask;
    inRange(image, data.colop, data.color, mask);

    if (data.softness > 0) {cv::blur(mask, mask, cv::Size(data.softness, data.softness));}

    image.setTo(Scalar(0, 0, 0), mask != 0);
    imgback.setTo(Scalar(0, 0, 0), mask == 0);


    Mat imgout = imgback + image;

    switch(waitKey(args.fps)) {
      case 'c':
        cout << "Key pressed: 'c'" << endl;
        break;
      case 27: // esc is pressed
        cout << "Key pressed: 'esc'. Stopping the video" << endl;
        return EXIT_FAILURE;
    }
    imshow(args.winName, imgout);
  }

  return EXIT_SUCCESS;
}
