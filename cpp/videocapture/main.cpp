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
#include "../api/ocvui.hpp"
#include "../api/ocvio.hpp"
#include "../api/text.hpp"




int main(int, char**) {

  ui::StyleLight style;

  cv::Ptr<cv::freetype::FreeType2> ft2r;
  ft2r = cv::freetype::createFreeType2();
  ft2r->loadFontData(style.fontr, 0);

  cv::dnn::Net net = cv::dnn::readNetFromCaffe(
    "/home/oa/Dropbox/code/oa/resources/ml/models/deploy.prototxt",
    "/home/oa/Dropbox/code/oa/resources/ml/models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
  );
  // Model parameters
  cv::Size inwh(256, 256);
  cv::Scalar mean(104, 117, 123);
  int conf_treshold(75);

  std::string winname = "Camera Preview";
	cv::namedWindow(winname, cv::WINDOW_NORMAL);
  cv::Mat frame;

  //--- INITIALIZE VIDEOCAPTURE
  cv::VideoCapture camera(0, cv::CAP_ANY);
  io::setup_video_capture(camera);
  std::array<int,2> res = io::get_resolution(camera);

  std::cout << "Camera width: " + std::to_string(res[0]) << std::endl;
  std::cout << "Camera height: " + std::to_string(res[1]) << std::endl;
  // std::cout << "Camera fps: " + std::to_string(int(camera.get(cv::CAP_PROP_FPS))) << std::endl;


  ui::Draw uidraw(net);
  // camera.open(0, cv::CAP_ANY);
  // if (!camera.isOpened()) {
  //   std::cerr << "ERROR! Unable to open camera\n";
  //   return -1;as
  // }
  //--- GRAB AND WRITE LOOP
  std::cout << "Start grabbing" << std::endl << "Press any key to terminate" << std::endl;
  while (cv::waitKey(1) != 27) {
    camera.read(frame);
    if (frame.empty()) {break;}
    cv::flip(frame, frame, 1);

    uidraw.set_image(frame);
    net.setInput(cv::dnn::blobFromImage(frame, 1.0, inwh, mean, false, false));
    // Run the forward pass
    std::vector<cv::Mat> outs;
    net.forward(outs);

    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    float confThreshold(0.75);
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    if (outLayerType == "DetectionOutput") {
      // Network produces output blob with a shape 1x1xNx7 where N is a number of
      // detections and an every detection is a vector of values
      // [batchId, classId, confidence, left, top, right, bottom]
      CV_Assert(outs.size() > 0);
      for (size_t k = 0; k < outs.size(); k++)
      {
        float* data = (float*)outs[k].data;
        for (size_t i = 0; i < outs[k].total(); i += 7)
        {
          float confidence = data[i + 2];
          if (confidence > confThreshold)
          {
            int left   = (int)data[i + 3];
            int top    = (int)data[i + 4];
            int right  = (int)data[i + 5];
            int bottom = (int)data[i + 6];
            int width  = right - left + 1;
            int height = bottom - top + 1;
            if (width <= 2 || height <= 2)
            {
              left   = (int)(data[i + 3] * frame.cols);
              top    = (int)(data[i + 4] * frame.rows);
              right  = (int)(data[i + 5] * frame.cols);
              bottom = (int)(data[i + 6] * frame.rows);
              width  = right - left + 1;
              height = bottom - top + 1;
            }
            classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
          }
        }
      }
    }


    for (size_t idx = 0; idx < boxes.size(); ++idx) {
      cv::Rect box = boxes[idx];
      // ui::BBox bbox(left, top, right, bottom);
      // ui::DBox dbox(box.x, box.y, box.x+box.width, box.y+box.height);
      ui::DBox dbox(boxes[idx].x, boxes[idx].y, boxes[idx].x+boxes[idx].width, boxes[idx].y+boxes[idx].height);

      uidraw.dbox_outline(uidraw.imsh, dbox, style.cola, 128);
      uidraw.dbox_frame(uidraw.imsh, dbox, style.cola, 0.1, 128);

      // ft2r->putText(frame, "left", dbox.left, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "top", dbox.top, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "right", dbox.right, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "bottom", dbox.bottom, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "center", dbox.center, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "lefttop", dbox.lefttop, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "righttop", dbox.righttop, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "rightbottom", dbox.rightbottom, 18, style.cola, -1, cv::LINE_AA, true);
      // ft2r->putText(frame, "leftbottom", dbox.leftbottom, 18, style.cola, -1, cv::LINE_AA, true);
    }

    uidraw.text(
      cv::format("Inference time: %.2f ms", ui::get_inference_time(net, "ms")),
      cv::Point(50, 50), style.cola, 18, 8, 128, true
    );

    // Show live and wait for a key with timeout long enough to show images
    cv::imshow(winname, frame);
  };

  // the camera will be deinitialized automatically in VideoCapture destructor
  return 0;
}
