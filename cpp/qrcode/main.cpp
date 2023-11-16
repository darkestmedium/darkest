// #pragma once

// System includes
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/dnn.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/videoio.hpp>

// Custom includes
#include "../api/ocvui.hpp"
#include "../api/ocvio.hpp"
#include "../api/text.hpp"




int main(int, char**) {
  std::string imgPath = "/home/oa/Dropbox/code/oa/resources/images/IDCard-Satya.png";

  cv::Mat img = cv::imread(imgPath, 1);
  std::cout << img.size().height << " " << img.size().width << std::endl;

  cv::Mat bbox, rectifiedImage;

  cv::QRCodeDetector qrDecoder;

  std::string opencvData = qrDecoder.detectAndDecode(img, bbox, rectifiedImage);

  // Check if a QR Code has been detected
  if(opencvData.length()>0)
    std::cout << "QR Code Detected" << std::endl;
  else
    std::cout << "QR Code NOT Detected" << std::endl;


  int n = bbox.rows;
  for(int i = 0 ; i < n ; i++) {
    cv::line(
      img, 
      cv::Point(bbox.at<float>(i,0),bbox.at<float>(i,1)), 
      cv::Point(bbox.at<float>((i+1) % n,0), bbox.at<float>((i+1) % n,1)), 
      cv::Scalar(1,0,0), 3
    );
  }

  std::cout << bbox << std::endl;

  std::cout << "Decoded Data : " << opencvData << std::endl;

  std::string resultImagePath = "home/oa/Dropbox/code/oa/QRCode-Output.png";
  cv::imwrite(resultImagePath, img);


  cv::imshow("qrDecoder", img);
  cv::waitKey(0);
}