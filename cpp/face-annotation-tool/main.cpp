// Import packages
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
 
//Using namespace to nullify use of cv::function(); syntax
using namespace cv;
using namespace std;
 
// Points to store the bounding box coordinates
Point top_left_corner, bottom_right_corner;
// image image
Mat image;




// function which will be called on mouse input
void drawRectangle(int action, int x, int y, int flags, void *userdata)
{
  // Mark the top left corner when left mouse button is pressed
  if( action == EVENT_LBUTTONDOWN )
  {
    top_left_corner = Point(x,y);
  }
  // When left mouse button is released, mark bottom right corner
  else if( action == EVENT_LBUTTONUP)
  {
    bottom_right_corner = Point(x,y);
    // Draw rectangle
    rectangle(image, top_left_corner, bottom_right_corner, Scalar(0,255,0), 2, 8 );
    // Display image
    imshow("Window", image);
    imwrite("face.png", temp(Range(topleft.y,bottomright.y),Range(topleft.x,bottomright.x)));
  }
   
}
 
// Main function
int main() {
  image = imread("/home/oa/Dropbox/code/darkest/resources/images/underwater.png");
  // Make a temporary image, which will be used to clear the image
  Mat temp = image.clone();
  // Create a named window
  namedWindow("Window");
  // highgui function called when mouse events occur
  setMouseCallback("Window", drawRectangle);
 
  int k=0;
  // loop until q character is pressed
  while(k!=27) {
    imshow("Window", image);
    putText(image, "Choose center, and drag, Press ESC to exit and c to clear", Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2);
    k= waitKey(0);
    // If c is pressed, clear the window, using the dummy image
    if(k == 99)
    {
      temp.copyTo(image);
    }
  }
  destroyAllWindows();
  return 0;
}