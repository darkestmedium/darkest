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

  std::string imgPath = "/home/oa/Dropbox/code/darkest/resources/images/underwater.png";

  cv::Mat img = cv::imread(imgPath, 1);
  std::cout << img.size().height << " " << img.size().width << std::endl;

  cv::imshow("Image", img);
}