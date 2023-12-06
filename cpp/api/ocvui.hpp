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



using namespace std;
using namespace cv;




struct DetectionBox {
  /* Detection box struct. Holds and calculates all the grid points.

  x, y, width, height;
  left, top, right, bottom
  pair<int, int> center = getCenter(bbox);

  */
  Point lefttop, righttop, rightbottom, leftbottom, left, top, right, bottom, center;
  vector<Point> points;

  // Constructors
  DetectionBox(int left, int top, int right, int bottom)
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
  ~DetectionBox() {};

  void points_to_array() {
    points[0]=lefttop;
    points[1]=righttop;
    points[2]=rightbottom;
    points[3]=leftbottom;
    points[4]=left;
    points[5]=top;
    points[6]=right;
    points[7]=bottom;
    points[8]=center;
  }
};


struct ScalingOCVUi {
  int thicka = 1;
  int thickb = 2;
  int thickc = 4;
};


struct StyleOCVUi {
  Scalar cola;
  Scalar colb;
  Scalar colc;
  Scalar cold;
  string fontr;
  string fontb;
  string fontl;
};


struct StyleLight : StyleOCVUi {
  // Constructor
  StyleLight()
    : StyleOCVUi()
    , cola{255, 255, 255}  // almost white
    , colb{225, 225, 225}  // light grey
    , colc{35, 35, 35}     // dark grey
    , cold{0, 0, 0}        // black
    , fontl("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Light.ttf")
    , fontr("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Regular.ttf")
    , fontb("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Bold.ttf")
  {};
  // Destructors
  ~StyleLight() {};

  // Color
  Scalar cola;  // almost white;
  Scalar colb;  // light grey;
  Scalar colc;  // dark grey;
  Scalar cold;  // black;

  // Fonts
  string fontl;
  string fontr;
  string fontb;

};




class Draw {
  /*Class for drawing annotations with the opencv framework.

    he origin, (0, 0), is located at the top-left of the image. OpenCV images are zero-indexed,
    where the x-values go left-to-right (column number) and y-values go top-to-bottom (row number).

    */
public:

  // Constructors
  Draw(Mat img, dnn::Net dnn)
    : imcv(img)
    , dnn(dnn)
  {
    ft2l->loadFontData(style.fontl, 0);
    ft2r->loadFontData(style.fontr, 0);
    ft2b->loadFontData(style.fontb, 0);
  };
  Draw(dnn::Net dnn)
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
  Mat imcv;
  Mat imsh;
  Mat imtxt;
  Mat imout;

  Size imsize;

  int thicka = 2;
  int thickb = thicka*2;
  int thickc = thicka*4;

  dnn::Net dnn;
  StyleLight style;

  // Freetype Classes
  Ptr<freetype::FreeType2> ft2l = freetype::createFreeType2();
  Ptr<freetype::FreeType2> ft2r = freetype::createFreeType2();
  Ptr<freetype::FreeType2> ft2b = freetype::createFreeType2();


  void set_image(Mat &img) {
    imcv = img;
    imsize = img.size();
    imsh = Mat(imcv.size(), CV_8UC4, Scalar(255, 255, 255, 0));
    imtxt = Mat(imcv.size(), CV_8UC4, Scalar(255, 255, 255, 0));
    imout = Mat(imcv.size(), CV_8UC3, Scalar(255, 0, 0));
  };


  double get_dbox_scaled_edge(DetectionBox& dbox, double scale) {
    return double(min((dbox.right.x-dbox.left.x), (dbox.top.y-dbox.bottom.y))) * scale;
  };


  void dbox_outline(DetectionBox& dbox, Scalar rgb, int opacity) {
    /* Draw a rectengular outline for the given bbox.
    */
    Scalar color(rgb[0], rgb[1], rgb[2], opacity);
    rectangle(imsh, dbox.lefttop, dbox.rightbottom, color, thicka);
  };


  void dbox_frame(DetectionBox& dbox, Scalar rgb, int opacity=127, double scale=0.1) {
    // left, top, right, bottom
    Scalar color(rgb[0], rgb[1], rgb[2], opacity);
    double edgs = get_dbox_scaled_edge(dbox, scale);
    // left top
    line(imsh, dbox.lefttop, Point(dbox.lefttop.x, dbox.lefttop.y-edgs), color, thickb);
    line(imsh, dbox.lefttop, Point(dbox.lefttop.x-edgs, dbox.lefttop.y), color, thickb);
    // right top
    line(imsh, dbox.righttop, Point(dbox.righttop.x, dbox.righttop.y-edgs), color, thickb);
    line(imsh, dbox.righttop, Point(dbox.righttop.x+edgs, dbox.righttop.y), color, thickb);
    // right bottom
    line(imsh, dbox.rightbottom, Point(dbox.rightbottom.x, dbox.rightbottom.y+edgs), color, thickb);
    line(imsh, dbox.rightbottom, Point(dbox.rightbottom.x+edgs, dbox.rightbottom.y), color, thickb);
    // left bottom
    line(imsh, dbox.leftbottom, Point(dbox.leftbottom.x, dbox.leftbottom.y+edgs), color, thickb);
    line(imsh, dbox.leftbottom, Point(dbox.leftbottom.x-edgs, dbox.leftbottom.y), color, thickb);
  };


  void dbox_contours() {
    cout << "draw contour" << endl;
  };


  Size get_text_size(string text, int fonth, int padding=8) {
    return ft2r->getTextSize(text, fonth+padding, -1, 0);
  }


  void text(string text, Point pos, Scalar rgb, int fonth=18, int padding=8, int opacity=255, bool draw_bbox=true) {
    Scalar color(rgb[0], rgb[1], rgb[2], opacity);
    Size txtwh = get_text_size(text, fonth, padding);
    if (draw_bbox == true) {
      rectangle(imsh, pos, Point(pos.x+txtwh.width, pos.y+txtwh.height), color, -1);
    }
    ft2r->putText(imtxt, text, pos, fonth, style.cold, -1, LINE_AA, false);
  };


  // Static kind of @classmethod /
  Mat combine() {
    /* The best way to access multi channel array with the c++ api is by creating a pointer to a specific row using the ptr method.

    For example;
      type elem = matrix.ptr<type>(i)[N~c~*j+c]

    Where:
      type: the datatype(float, int, char ect..)
      i: row you're interested in
      Nc: the number of channels
      j: the column you're interested in
      c: the column you're interested in(0-3)

    For information on other c->c++ conversion check out this link: Source

    */
    imcv.convertTo(imcv, CV_32FC3);
    imsh.convertTo(imsh, CV_32FC4);
    imtxt.convertTo(imtxt, CV_32FC4);
    imout.convertTo(imout, CV_32FC3);

    Mat imsh_mask;
    inRange(imsh, Scalar(0,0,0,1), Scalar(255,255,255,255), imsh_mask);

    for(int y=0; y<imcv.size().height; y++) {
      for(int x=0; x<imcv.size().width; x++) {
        for(int c=0; c<imcv.channels(); c++) {
          imout.at<Vec3f>(y,x)[c] = (
            (imcv.at<Vec3f>(y,x)[c] * (1.0-imsh.at<Vec4f>(y,x)[3]))
            +(imsh.at<Vec4f>(y,x)[c] * imsh.at<Vec4f>(y,x)[3]) * (1.0-imtxt.at<Vec4f>(y,x)[3])
            +(imtxt.at<Vec4f>(y,x)[c] * imtxt.at<Vec4f>(y,x)[3])
          );
        }
      }
    }
    
    imout.convertTo(imout, CV_8UC3);
    return imout;
  }
};




// namespace ui inline methods
inline double get_inference_time(dnn::Net dnn, string unit="ms") {
  vector<double> timmings;
  dnn.getPerfProfile(timmings);
  double timer = timmings[0];
  double frequency = getTickFrequency() / 1000;
  float time_eval;
  if (unit == "ms") {
    time_eval = dnn.getPerfProfile(timmings) / frequency;
  }
  if (unit == "fps") {
    time_eval = frequency / timer;
  }
  return time_eval;
};


