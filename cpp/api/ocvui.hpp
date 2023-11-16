#pragma once

// System includes
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/freetype.hpp>






namespace ui {


  struct DBox {
    /* Detection box struct. Holds and calculates all the grid points.

    x, y, width, height;
    left, top, right, bottom
    std::pair<int, int> center = getCenter(bbox);

    */

    cv::Point lefttop, righttop, rightbottom, leftbottom, left, top, right, bottom, center;
    // Constructors
    DBox(int left, int top, int right, int bottom)
      : lefttop(left, top)
      , righttop(right, top)
      , rightbottom(right, bottom)
      , leftbottom(left, bottom)
      , left(left, (top+bottom)/2)
      , top((left+right)/2, top)
      , right(right, (top+bottom)/2)
      , bottom((left+right)/2, bottom)
      , center((left+right)/2, (top+bottom)/2)
    {};
    // Destructors
    ~DBox() {};
  
    // cv::Point get_bbox_enter(int left, int top, int right, int bottom) {return cv::Point((left+right/2), (top+bottom/2));}
  };


  struct ScalingOCVUi {
    int thicka = 1;
    int thickb = 2;
    int thickc = 4;

  };


  struct StyleOCVUi {
    cv::Scalar cola;
    cv::Scalar colb;
    cv::Scalar colc;
    cv::Scalar cold;
    std::string fontr;
    std::string fontb;
    std::string fontl;
  };


  struct StyleLight : StyleOCVUi {
    // Constructor
    // StyleLight()
    //   : StyleOCVUi()
    //   , cola{255, 255, 255}  // almost white
    //   , colb{225, 225, 225}  // light grey
    //   , colc{35, 35, 35}     // dark grey
    //   , cold{0, 0, 0}        // black
    //   , fontl("/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Light.ttf")
    //   , fontr("/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Regular.ttf")
    //   , fontb("/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Bold.ttf")
    //   , ft2l->loadFontData(fontrl)
    // {};
    // // Destructors
    // ~StyleLight() {};

    // Color
    cv::Scalar cola = {255, 255, 255};  // almost white;
    cv::Scalar colb = {225, 225, 225} ; // light grey;
    cv::Scalar colc = {35, 35, 35};     // dark grey;
    cv::Scalar cold = {0, 0, 0};        // black;

    // Fonts
    std::string fontl = "/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Light.ttf";
    std::string fontr = "/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Regular.ttf";
    std::string fontb = "/home/oa/Dropbox/code/oa/resources/fonts/intel/IntelOneMono-Bold.ttf";

  };




  class Draw {
    /*Class for drawing annotations with the opencv framework.

      he origin, (0, 0), is located at the top-left of the image. OpenCV images are zero-indexed,
      where the x-values go left-to-right (column number) and y-values go top-to-bottom (row number).

     */
  public:

    // Constructors
    Draw(cv::Mat img, cv::dnn::Net dnn)
      : imcv(img)
      , dnn(dnn)
    {
      ft2l->loadFontData(style.fontl, 0);
      ft2r->loadFontData(style.fontr, 0);
      ft2b->loadFontData(style.fontb, 0);
    };
    Draw(cv::dnn::Net dnn)
      : dnn(dnn)
    {
      ft2l->loadFontData(style.fontl, 0);
      ft2r->loadFontData(style.fontr, 0);
      ft2b->loadFontData(style.fontb, 0);
    };
    Draw() {
      ft2l->loadFontData(style.fontl, 0);
      ft2r->loadFontData(style.fontr, 0);
      ft2b->loadFontData(style.fontb, 0);
    };
    // Destructors
    ~Draw() {delete ft2l, ft2r, ft2b;}

    // Image Layers -> input image, shapes, text, combined out image
    cv::Mat imcv;
    cv::Mat imsh;
    cv::Mat imtxt;
    cv::Mat imout;

    int imwidth;
    int imheight;

    int thicka = 1;
    int thickb = thicka*2;
    int thickc = thicka*4;

    cv::dnn::Net dnn;
    StyleLight style;

    // Freetype Classes
    cv::Ptr<cv::freetype::FreeType2> ft2l = cv::freetype::createFreeType2();
    cv::Ptr<cv::freetype::FreeType2> ft2r = cv::freetype::createFreeType2();
    cv::Ptr<cv::freetype::FreeType2> ft2b = cv::freetype::createFreeType2();


    void set_image(cv::Mat &img) {
      // img.convertTo(imcv, CV_32FC3, 1.0/255);
      imcv = img;
      imheight = imcv.size[0];
      imwidth = imcv.size[1];
      imsh = cv::Mat(imcv.size(), CV_8UC3, cv::Scalar(0, 0, 0, 0));
      imtxt = cv::Mat(imcv.size(), CV_8UC3, cv::Scalar(0, 0, 0, 0));
      imout = cv::Mat(imcv.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    };


    double get_dbox_scaled_edge(ui::DBox& dbox, double scale=0.1) {
      return double(std::min((dbox.right.x-dbox.left.x), (dbox.top.y-dbox.bottom.y))) * scale;
    };


    void dbox_outline(cv::Mat& img, ui::DBox& dbox, cv::Scalar rgb, int opacity) {
      /* Draw a rectengular outline for the given bbox. */
      cv::Scalar color(rgb[0], rgb[1], rgb[2], opacity);
      cv::rectangle(img, dbox.lefttop, dbox.rightbottom, color, thicka);
    };


    void dbox_frame(cv::Mat& img, ui::DBox& dbox, cv::Scalar rgb, double scale=0.1, int opacity=255) {
      // left, top, right, bottom
      cv::Scalar color(rgb[0], rgb[1], rgb[2], opacity);
      double edgs = get_dbox_scaled_edge(dbox, scale);
      // left top
      cv::line(img, dbox.lefttop, cv::Point(dbox.lefttop.x, dbox.lefttop.y-edgs), color, thickb);
      cv::line(img, dbox.lefttop, cv::Point(dbox.lefttop.x-edgs, dbox.lefttop.y), color, thickb);
      // right top
      cv::line(img, dbox.righttop, cv::Point(dbox.righttop.x, dbox.righttop.y-edgs), color, thickb);
      cv::line(img, dbox.righttop, cv::Point(dbox.righttop.x+edgs, dbox.righttop.y), color, thickb);
      // right bottom
      cv::line(img, dbox.rightbottom, cv::Point(dbox.rightbottom.x, dbox.rightbottom.y+edgs), color, thickb);
      cv::line(img, dbox.rightbottom, cv::Point(dbox.rightbottom.x+edgs, dbox.rightbottom.y), color, thickb);
      // left bottom
      cv::line(img, dbox.leftbottom, cv::Point(dbox.leftbottom.x, dbox.leftbottom.y+edgs), color, thickb);
      cv::line(img, dbox.leftbottom, cv::Point(dbox.leftbottom.x-edgs, dbox.leftbottom.y), color, thickb);
    };


    cv::Size get_text_size(std::string text, int fonth, int padding=8) {
      return ft2r->getTextSize(text, fonth+padding, -1, 0);
    }


    void text(std::string text, cv::Point pos, cv::Scalar rgb, int fonth=18, int padding=8, int opacity=255, bool draw_bbox=true) {
      cv::Scalar color(rgb[0], rgb[1], rgb[2], opacity);
      cv::Size txtwh = get_text_size(text, fonth, padding);
      if (draw_bbox == true) {
        cv::rectangle(imsh, pos, cv::Point(pos.x+txtwh.width, pos.y+txtwh.height), color, -1);
      }
      ft2r->putText(imtxt, text, pos, fonth, style.cold, -1, cv::LINE_AA, false);
    };


    // Static kind of @classmethod /
    cv::Mat combine() {
      /* The best way to access multi channel array with the c++ api is by creating a pointer to a specific row using the ptr method.

      For example;
        type elem = matrix.ptr<type>(i)[N~c~*j+c]

      where:
        type: the datatype(float, int, char ect..)
        i: row you're interested in
        Nc: the number of channels
        j: the column you're interested in
        c: the column you're interested in(0-3)

      For information on other c->c++ conversion check out this link: Source

      */


      cv::Mat imsh_mask;
      cv::inRange(imsh, cv::Scalar(0,0,0,1), cv::Scalar(255,255,255,255), imsh_mask);
      // imout = imsh_mask;
      imout = imcv;


      // imcv.convertTo(imcv, CV_8UC3);
      // int numberOfPixels = imcv.rows * imcv.cols * imcv.channels();
      // Get floating point pointers to the data matrices
      // char8_t* pimcv = reinterpret_cast<char8_t*>(imcv.data);
      // char8_t* pimsh = reinterpret_cast<char8_t*>(imsh.data);
      // char8_t* pimtxt = reinterpret_cast<char8_t*>(imtxt.data);
      // char8_t* pimout = reinterpret_cast<char8_t*>(imout.data);
      // Assuming imsh, imtxt, imcv are cv::Mat objects

      // masksh.convertTo(masksh, CV_8UC1, 1.0/255);

      // for (int channel = 0; channel < 3; ++channel) {
      //   imout.col(channel) =
      //     ((imcv.col(channel) * (1.0 - masks))
      //     +(imsh.col(channel) * masks) * (1.0 - maskt))
      //     +(imtxt.col(channel) * maskt);
      // }
      // for (int y=0; y<imcv.rows; y++) {
      //   for (int x=0; x<imcv.cols; x++) {
      //     int idxrgb = x * 3 + y*imcv.step;  // Blue
      //     int idxa = x + y*imcv.step;

      //     int B = pimcv[idxrgb];
      //     int G = pimcv[idxrgb+1];
      //     int R = pimcv[idxrgb+2];

      //     // int idxsh = x * 4 + y*imcv.step;
      //     int Bsh = pimsh[idxrgb];
      //     int Gsh = pimsh[idxrgb+1];
      //     int Rsh = pimsh[idxrgb+2];
      //     // int Ash = pimsh[idxsh+3];

      //     pimout[idxrgb] = B;
      //     pimout[idxrgb+1] = G;
      //     pimout[idxrgb+2] = R;
      //   }
      // }

      // double alpha = 1.0; /*< Simple contrast control */
      // int beta = 0;       /*< Simple brightness control */

      // for(int y = 0; y < imcv.rows; y++) {
      //   for(int x = 0; x < imcv.cols; x++) {
      //     for(int c = 0; c < imcv.channels(); c++) {
      //       imout.at<cv::Vec3b>(y,x)[c] = cv::saturate_cast<uchar>(alpha*imcv.at<cv::Vec3b>(y,x)[c] + beta);
      //     }
      //   }
      // }

      


      // imout.convertTo(imout, CV_8UC3);
      // imcv.convertTo(imcv, CV_8UC4);
      // cv::Mat imshmask;
      // cv::extractChannel(imsh, imshmask, 3);
      // imshmask.convertTo(imshmask, CV_8UC4);

      // alphaBlend(imsh, imcv, imshmask, imout);

      return imout;
    }


    // void alphaBlend(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& outImage) {
    //   // Find number of pixels.
    //   int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    //   // Get floating point pointers to the data matrices
    //   uint8_t* fptr = reinterpret_cast<uint8_t*>(foreground.data);
    //   uint8_t* bptr = reinterpret_cast<uint8_t*>(background.data);
    //   uint8_t* aptr = reinterpret_cast<uint8_t*>(alpha.data);
    //   uint8_t* outImagePtr = reinterpret_cast<uint8_t*>(outImage.data);

    //   // Loop over all pixesl ONCE
    //   for(
    //     int i = 0;
    //     i < numberOfPixels;
    //     i++, outImagePtr++, fptr++, aptr++, bptr++
    //   )
    //   {
    //     *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
    //   }
    // };
  };


  // namespace ui inline methods
  inline double get_inference_time(cv::dnn::Net dnn, std::string unit="ms") {
    std::vector<double> timmings;
    dnn.getPerfProfile(timmings);
    double timer = timmings[0];
    double frequency = cv::getTickFrequency() / 1000;
    float time_eval;
    if (unit == "ms") {
      time_eval = dnn.getPerfProfile(timmings) / frequency;
    }
    if (unit == "fps") {
      time_eval = frequency / timer;
    }
    return time_eval;
  };

};  // namespace ui

