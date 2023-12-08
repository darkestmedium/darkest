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
  */
  Point lefttop, righttop, rightbottom, leftbottom, left, top, right, bottom, center;
  array<Point,9> points;

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
  {points_to_array();}
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




struct StyleLight {
  // Constructors
  StyleLight()
    : cola{255, 255, 255}  // almost white
    , colb{225, 225, 225}  // light grey
    , colc{35, 35, 35}     // dark grey
    , cold{0, 0, 0}        // black
    , fontl("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Light.ttf")
    , fontr("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Regular.ttf")
    , fontb("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Bold.ttf")
  {};
  // Destructors
  ~StyleLight() {};

  Scalar cola;
  Scalar colb;
  Scalar colc;
  Scalar cold;
  string fontr;
  string fontb;
  string fontl;
};


struct StyleDark {
  // Constructors
  StyleDark()
    : cold{0, 0, 0}        // black
    , colc{39, 39, 39}     // dark grey
    , colb{225, 225, 225}  // light grey
    , cola{245, 245, 245}  // almost white
    , fontl("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Light.ttf")
    , fontr("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Regular.ttf")
    , fontb("/home/ccpcpp/Dropbox/code/darkest/resources/fonts/intel/IntelOneMono-Bold.ttf")
  {};
  // Destructors
  ~StyleDark() {};

  Scalar cola;
  Scalar colb;
  Scalar colc;
  Scalar cold;
  string fontr;
  string fontb;
  string fontl;
};




class Draw {
  /*Class for drawing annotations with the opencv framework.

    he origin, (0, 0), is located at the top-left of the image. OpenCV images are zero-indexed,
    where the x-values go left-to-right (column number) and y-values go top-to-bottom (row number).

    */
public:
  // Constructors
  Draw(Mat image, dnn::Net dnn, string color="light")
    : base(image)
    , dnn(dnn)
  { 
    set_style(color);
    ft2l->loadFontData(style.fontl, 0);
    ft2r->loadFontData(style.fontr, 0);
    ft2b->loadFontData(style.fontb, 0);
  };
  Draw(dnn::Net dnn, string color="light")
    : dnn(dnn)
  {
    set_style(color);
    ft2l->loadFontData(style.fontl, 0);
    ft2r->loadFontData(style.fontr, 0);
    ft2b->loadFontData(style.fontb, 0);
  };
  Draw(string color="light") {
    set_style(color);
    ft2l->loadFontData(style.fontl, 0);
    ft2r->loadFontData(style.fontr, 0);
    ft2b->loadFontData(style.fontb, 0);
  };
  // Destructors
  ~Draw() {delete ft2l, ft2r, ft2b;}

  // Image Layers -> base, shapes, typo, composite 
  Mat base;
  Mat shapes;
  Mat typo;
  Mat composite;

  Size frame_size;

  int thicka = 2;
  int thickb = thicka*2;
  int thickc = thicka*4;

  int heading=16;
  int body=12;

  dnn::Net dnn;
  StyleLight style;

  // Freetype Classes
  Ptr<freetype::FreeType2> ft2l = freetype::createFreeType2();
  Ptr<freetype::FreeType2> ft2r = freetype::createFreeType2();
  Ptr<freetype::FreeType2> ft2b = freetype::createFreeType2();


  void set_image(Mat &image) {
    base = image;
    shapes = Mat(base.size(), CV_8UC4, Scalar(255, 255, 255, 0));
    typo = Mat(base.size(), CV_8UC4, Scalar(255, 255, 255, 0));
    composite = Mat(base.size(), CV_8UC3, Scalar(0, 0, 0));
  };


  void set_style(string color="light") {
    if (color == "light") {
      StyleLight style;
    }
    if (color == "dark") {
      StyleDark style;
    }
  }


  float get_dbox_scaled_edge(DetectionBox& dbox, float scale=0.1) {
    return float(min((dbox.right.x-dbox.left.x), (dbox.top.y-dbox.bottom.y))) * scale;
  };


  void outline(DetectionBox& dbox, int opacity=127) {
    /* Draw a rectengular outline for the given bbox.
    */
    Scalar color(style.cola[0], style.cola[1], style.cola[2], opacity);
    rectangle(shapes, dbox.lefttop, dbox.rightbottom, color, thicka);
  };


  void frame(DetectionBox& dbox, int opacity=127, float scale=0.1) {
    Scalar color(style.cola[0], style.cola[1], style.cola[2], opacity);
    float edge = get_dbox_scaled_edge(dbox, scale);
    // Left top
    line(shapes, dbox.lefttop, Point(dbox.lefttop.x, dbox.lefttop.y-edge), color, thickb);
    line(shapes, dbox.lefttop, Point(dbox.lefttop.x-edge, dbox.lefttop.y), color, thickb);
    // Right top
    line(shapes, dbox.righttop, Point(dbox.righttop.x, dbox.righttop.y-edge), color, thickb);
    line(shapes, dbox.righttop, Point(dbox.righttop.x+edge, dbox.righttop.y), color, thickb);
    // Right bottom
    line(shapes, dbox.rightbottom, Point(dbox.rightbottom.x, dbox.rightbottom.y+edge), color, thickb);
    line(shapes, dbox.rightbottom, Point(dbox.rightbottom.x+edge, dbox.rightbottom.y), color, thickb);
    // Left bottom
    line(shapes, dbox.leftbottom, Point(dbox.leftbottom.x, dbox.leftbottom.y+edge), color, thickb);
    line(shapes, dbox.leftbottom, Point(dbox.leftbottom.x-edge, dbox.leftbottom.y), color, thickb);
  };


  void dbox_contours() {
    cout<<"draw contour"<<endl;
  };


  Size get_text_size(string text, int fonth=0, int padding=4) {
    if(fonth==0) {fonth = heading;}
    return ft2r->getTextSize(text, fonth+padding, -1, 0);
  }


  void text(string text, Point pxy, int fonth=0, int padding=4, string alignh="left", string alignv="above", int boxo=127, int txto=255, bool drawBox=true) {
    Scalar color(style.cola[0], style.cola[1], style.cola[2], boxo);
    if(fonth == 0) {fonth = heading;}
    Size twh = get_text_size(text, fonth, padding);

    Point bxy, bwh, txy;
    int padh = int(padding*0.5);
    Size twhh = Size(twh.width*0.5, twh.height*0.5);

    // Quick Python port prolly needs optimization
    if (alignh == "left" or alignv == "above") {
      bxy = Point(pxy.x, pxy.y-twh.height-padding);
      bwh = Point(pxy.x+twh.width+padding, pxy.y);
      txy = Point(bxy.x+padh, bwh.y-padh);
    }
    if(alignh == "center") {
      bxy = Point(pxy.x-twhh.width-padh, pxy.y-twh.height-padding);
      bwh = Point(pxy.x+twhh.width+padh, pxy.y);
      txy = Point(bxy.x+padh, bwh.y-padh);
    } 
    if(alignh == "right") {
      bxy = Point(pxy.x-twh.width-padding, pxy.y);
      bwh = Point(pxy.x, pxy.y-twh.height-padding);
      txy = Point(bxy.x+padh, bxy.y-padh);
    }
    if(alignv == "center") {
      bxy = Point(bxy.x, bxy.y + twhh.height + padh);
      bwh = Point(bwh.x, bwh.y + twhh.height + padh);
      txy = Point(txy.x, txy.y + twhh.height + padh);
    }
    if(alignv == "below") {
      bxy = Point(bxy.x, bxy.y + twh.height + padding);
      bwh = Point(bwh.x, bwh.y + twh.height + padding);
      txy = Point(txy.x, txy.y + twh.height + padding);
    }

    if (drawBox == true) {rectangle(shapes, bxy, bwh, color, -1);}
    ft2r->putText(typo, text, txy, fonth, Scalar(style.cold[0], style.cold[1], style.cold[2], txto), -1, LINE_AA, true);
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
    base.convertTo(base, CV_32FC3);
    shapes.convertTo(shapes, CV_32FC4);
    typo.convertTo(typo, CV_32FC4);
    composite.convertTo(composite, CV_32FC3);

    Mat shapes_mask;
    extractChannel(shapes, shapes_mask, 3);
    shapes_mask *= 0.00392156863f;
    // shapes_mask /= 255.f;

    Mat typo_mash;
    extractChannel(typo, typo_mash, 3);
    typo_mash *= 0.00392156863f;
    // typo_mash /= 255.f;

    // Blend composites
    // optimize later? https://learnopencv.com/alpha-blending-using-opencv-cpp-python/ 
    for(int y=0; y<base.size().height; y++) {
      for(int x=0; x<base.size().width; x++) {
        for(int c=0; c<base.channels(); c++) {
          composite.at<Vec3f>(y,x)[c] = (
            (base.at<Vec3f>(y,x)[c] * (1.0-shapes_mask.at<float>(y,x)))
            +(shapes.at<Vec4f>(y,x)[c] * shapes_mask.at<float>(y,x)) * (1.0-typo_mash.at<float>(y,x))
            +(typo.at<Vec4f>(y,x)[c] * typo_mash.at<float>(y,x))
          );
        }
      }
    }
    composite.convertTo(composite, CV_8UC3);
    return composite;
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


