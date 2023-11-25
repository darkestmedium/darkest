#pragma once

// System Includes
#include <string>
#include <vector>

// OpenCV includes
#include <opencv2/imgproc.hpp>




namespace io {
  /* Input & Output wrapper namespace. */


  inline void set_resolution(cv::VideoCapture &camera, int width=1280, int height=720) {
    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
  }


  inline std::array<int,2> get_resolution(cv::VideoCapture &camera) {
    std::array<int,2> resolution = {
      static_cast<int>(camera.get(cv::CAP_PROP_FRAME_WIDTH)),
      static_cast<int>(camera.get(cv::CAP_PROP_FRAME_HEIGHT))
    };
    return resolution;
  }


  inline void setup_video_capture(cv::VideoCapture &camera, int width=1280, int height=720) {
  // Force native camera resolution with mjpg on linux
    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    set_resolution(camera, width, height);
  }


  inline void query_maximum_resolution(cv::VideoCapture *camera, int& max_width, int& max_height) {
    // Save current resolution
    const int current_width  = static_cast<int>(camera->get(cv::CAP_PROP_FRAME_WIDTH));
    const int current_height = static_cast<int>(camera->get(cv::CAP_PROP_FRAME_HEIGHT));

    // Get maximum resolution
    camera->set(cv::CAP_PROP_FRAME_WIDTH,  16000);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, 16000);
    max_width  = static_cast<int>(camera->get(cv::CAP_PROP_FRAME_WIDTH));
    max_height = static_cast<int>(camera->get(cv::CAP_PROP_FRAME_HEIGHT));

    // Restore resolution
    camera->set(cv::CAP_PROP_FRAME_WIDTH,  current_width);
    camera->set(cv::CAP_PROP_FRAME_HEIGHT, current_height);
  }

}
