// System includes
#include <vector>
#include <iostream>
#include <fstream>

// OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/freetype.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>




// Using namespace to nullify use of cv::function(); syntax
using namespace cv;
using namespace std;




// Read the input image
// string imageName = + "images/dilation_example.jpg";
// Mat image = imread(imageName, IMREAD_COLOR);




Mat Ila(Mat& source, Mat& element, VideoWriter& writer) {
  int ksize = element.size().height;
  int height = source.size().height;
  int width  = source.size().width;

  int border = ksize/2;
  Mat paddedSource = Mat::zeros(Size(height + border*2, width + border*2), CV_8UC1);

  cout << ksize << endl;
  cout << height+border*2 << endl;
  cout << width+border*2 << endl;

  Mat bitOR;
  for (int h_i=border; h_i < height+border; h_i++) {
    for (int w_i=border; w_i < width+border; w_i++) {
      if (source.at<uchar>(h_i-border, w_i-border)) {
        bitwise_and(paddedSource(Range(h_i-border,h_i+border+1), Range(w_i-border,w_i+border+1)), element, bitOR);
        bitOR.copyTo(paddedSource(Range(h_i-border,h_i+border+1), Range(w_i-border,w_i+border+1)));
        cvtColor(paddedSource, paddedSource, 0);
        writer.write(paddedSource);
        waitKey(25);
      }
    }
  }
  return paddedSource*255;
}



Mat Ero(Mat& source, Mat element, VideoWriter& writer) {
  int ksize = element.size().height;
  int height = source.size().height;
  int width  = source.size().width;

  int border = ksize/2;
  Mat paddedSource = Mat::zeros(Size(height + border*2, width + border*2), CV_8UC1);

  Mat bitOR;
  for (int h_i=border; h_i < height + border; h_i++) {
    for (int w_i=border; w_i < width + border; w_i++) {
      if (source.at<uchar>(h_i-border, w_i-border)) {
        bitwise_or(paddedSource(Range(h_i-border,h_i+border+1), Range(w_i-border,w_i+border+1)), element, bitOR);
        bitOR.copyTo(paddedSource(Range(h_i-border,h_i+border+1), Range(w_i-border,w_i+border+1)));
      }
    }
  }

  return paddedSource*255;
}



// Main function
int main() {
  // Create a window to display results and set the flag to Autosize
  string windowName = "Show Image";
  namedWindow(windowName, WINDOW_NORMAL);

  // Create demo image
  Mat demoImage = Mat::zeros(Size(10,10), CV_8UC1);
  demoImage.at<uchar>(0,1) = 1;
  demoImage.at<uchar>(9,0) = 1;
  demoImage.at<uchar>(8,9) = 1;
  demoImage.at<uchar>(2,2) = 1;
  demoImage(Range(5,8), Range(5,8)).setTo(1);

  // Create element
  Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

  // Video capture
  VideoWriter writer("/home/oa/Downloads/dilation.avi", VideoWriter::fourcc('M','J','P','G'), 2, Size(10, 10));
  // writer.open("/home/oa/Downloads/dilation.avi", VideoWriter::fourcc('M','J','P','G'), 2, Size(10, 10));

  Mat outFrame(Ila(demoImage, element, writer));

  writer.release();

  // Display image
  imshow(windowName, outFrame);
  // waitKey(0);
  destroyAllWindows();
  return 0;
}