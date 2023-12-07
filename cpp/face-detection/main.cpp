// System includes
#include <iostream>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <math.h>


// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>


// Custom includes
#include "../api/argparse.hpp"
#include "../api/ocvui.hpp"
#include "../api/ocvio.hpp"
#include "../api/text.hpp"




using namespace std;
using namespace cv;




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
  string &filePath      = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/ml/models/deploy.prototxt");
  string &filePathDNN   = kwarg("fpd,filePathDNN", "Path to the dnn file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/ml/models/res10_300x300_ssd_iter_140000_fp16.caffemodel");
  int &width            = kwarg("w,width", "Stream width.").set_default(1280);
  int &height           = kwarg("h,height", "Stream height.").set_default(720);
  int &fps              = kwarg("fps,framerate", "Framerate.").set_default(30);
  int &camera           = kwarg("cam,camera", "Camera input - default is 0.").set_default(0);
  int &mirror           = kwarg("mir,mirror", "Mirror the camera input.").set_default(1);
  string &winName       = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
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

  dnn::Net net = dnn::readNetFromCaffe(args.filePath, args.filePathDNN );

  namedWindow(args.winName, WINDOW_NORMAL);
  Mat frame;
  int conf_treshold(75);

  //--- INITIALIZE VIDEOCAPTURE
  VideoCapture camera(args.camera);
  if(!camera.isOpened()) {cout << "Error opening video stream or file: " << args.camera << endl;}
  io::setup_video_capture(camera);

  Draw uidraw(net, "dark");

  //--- GRAB AND WRITE LOOP
  std::cout << "Start grabbing" << std::endl << "Press any key to terminate" << std::endl;
  while (camera.isOpened()) {
    camera.read(frame);
    if (frame.empty()) {break;}
    flip(frame, frame, args.mirror);
    uidraw.set_image(frame);

    net.setInput(dnn::blobFromImage(frame, 1.0, Size(256, 256), Scalar(104, 117, 123), false, false));

    // Run the forward pass
    std::vector<Mat> outs;
    net.forward(outs);

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    float confThreshold(0.75);
    std::vector<float> confidences;
    std::vector<Rect> boxes;

    if (outLayerType == "DetectionOutput") {
      // Network produces output blob with a shape 1x1xNx7 where N is a number of
      // detections and an every detection is a vector of values
      CV_Assert(outs.size() > 0);
      for (size_t k = 0; k < outs.size(); k++) {
        float* data = (float*)outs[k].data;
        for (size_t i = 0; i < outs[k].total(); i += 7) {
          float confidence = data[i+2];
          if (confidence > confThreshold) {
            int left   = int(data[i+3]);
            int top    = int(data[i+4]);
            int right  = int(data[i+5]);
            int bottom = int(data[i+6]);
            int width  = right-left+1;
            int height = bottom-top+1;
            if (width <= 2 || height <= 2) {
              left   = int(data[i+3]*frame.cols);
              top    = int(data[i+4]*frame.rows);
              right  = int(data[i+5]*frame.cols);
              bottom = int(data[i+6]*frame.rows);
              width  = right-left+1;
              height = bottom-top+1;
            }
            classIds.push_back(int((data[i + 1])-1));  // Skip 0th background class id.
            boxes.push_back(Rect(left, top, width, height));
            confidences.push_back(confidence);
          }
        }
      }
    }


    for (size_t idx = 0; idx < boxes.size(); ++idx) {
      DetectionBox dbox(boxes[idx].x, boxes[idx].y, boxes[idx].x+boxes[idx].width, boxes[idx].y+boxes[idx].height);
      uidraw.outline(dbox);
      uidraw.frame(dbox);


      uidraw.text("left top", dbox.lefttop, uidraw.heading, 4, "right", "below", 127, 255, true);
      uidraw.text("right top", dbox.righttop, uidraw.heading, 4, "left", "below", 127, 255, true);
      uidraw.text("right bottom", dbox.rightbottom, uidraw.heading, 4, "left", "above", 127, 255, true);
      uidraw.text("left bottom", dbox.leftbottom, uidraw.heading, 4, "right", "above", 127, 255, true);

      uidraw.text("left", dbox.left, uidraw.heading, 4, "right", "above", 127, 255, true);
      uidraw.text("right", dbox.right, uidraw.heading, 4, "left", "below", 127, 255, true);
    
      uidraw.text("top", dbox.top, uidraw.heading, 4, "center", "above", 127, 255, true);
      uidraw.text("bottom", dbox.bottom, uidraw.heading, 4, "center", "below", 127, 255, true);

      uidraw.text("center", dbox.center, uidraw.heading, 4, "center", "center", 127, 255, true);


      drawMarker(uidraw.shapes, dbox.center, Scalar(uidraw.style.cola[0], uidraw.style.cola[1], uidraw.style.cola[2], 127), MARKER_CROSS, uidraw.heading, uidraw.thicka, LINE_AA);

      // uidraw.ft2r->putText(uidraw.imcv, "left", dbox.left, 24, style.cola, -1, LINE_AA, true);
      // ft2r->putText(frame, "top", dbox.top, 18, style.cola, -1, LINE_AA, true);
      // ft2r->putText(frame, "right", dbox.right, 18, style.cola, -1, LINE_AA, true);
    }
    // string inference = format("Inference time: %.2f ms", get_inference_time(net, "ms"));
    // uidraw.text(
    //   inference,

    //   Point(50, 50), 18, 8, 128, false
    // );
    switch(waitKey(args.fps)) {
      case 'c':
        cout<<"Key pressed: 'c'"<<endl;
        break;
      case 27: // esc is pressed
        cout<<"Key pressed: 'esc'. Stopping the video"<<endl;
        return EXIT_FAILURE;
    }
    imshow(args.winName, uidraw.combine());
  };

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
