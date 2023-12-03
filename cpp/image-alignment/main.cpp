// System includes
#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"

// Custom includes
#include "../api/argparse.hpp"




using namespace std;
using namespace cv;




struct Syntax : public argparse::Args {
  std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/emir.jpg");
  std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
  bool &verbose         = flag("v,verbose", "Toggle verbose");
  bool &help            = flag("h,help", "Display usage");

  string commandName = "camera";

  void displayHelp() {
    cout <<
      "Command: "<<commandName<<"\n\n"
      "Flags:\n"
      "  --fp  --filePath (string):  Path to the file, ex. 'image.jpg'.\n"
      "  --wn  --winName (string):  Name of the window.\n"
      "  --v   --verbose (flag):  Toggle Verbose mode.\n\n"
      "Example usage :\n"
      "  "<<commandName<<" --filePath imgage1.jpg\n"
    << endl;
  };
};



int main(int argc, char* argv[]) {
  auto args = argparse::parse<Syntax>(argc, argv);

  if(args.help) {args.displayHelp(); return EXIT_FAILURE;}
  if(args.verbose) args.print();

  namedWindow(args.winName, WINDOW_NORMAL);

  Mat img = imread(args.filePath, IMREAD_GRAYSCALE);
  if (img.empty())  {
    cout<<"Can't read file '"<<args.filePath<<"'\n";
    return EXIT_FAILURE;
  }

  // Mat imageCopy = img.clone();
  // Find the width and height of the color image
  Size sz = img.size();
  int height = sz.height / 3;
  int width = sz.width;

  cout << sz << endl;

  // Extract the three channels from the gray scale image
  vector<Mat> channels;
  channels.push_back(img(Rect(0, 0,        width, height)));
  channels.push_back(img(Rect(0, height,   width, height))); 
  channels.push_back(img(Rect(0, 2*height, width, height)));

  Mat blue = channels[0];
  Mat green = channels[1];
  Mat red = channels[2];


  ///
  /// YOUR CODE HERE
  ///
  int MAX_FEATURES = 2048;
  float GOOD_MATCH_PERCENT = 0.1;

  ///
  /// YOUR CODE HERE
  ///
  // Initiate ORB detector
  Ptr<ORB> orb = ORB::create(MAX_FEATURES);

  vector<KeyPoint> keypointsBlue, keypointsGreen, keypointsRed;
  Mat descriptorsBlue, descriptorsGreen, descriptorsRed;
  orb->detectAndCompute(blue, Mat(), keypointsBlue, descriptorsBlue);
  orb->detectAndCompute(green, Mat(), keypointsGreen, descriptorsGreen);
  orb->detectAndCompute(red, Mat(), keypointsRed, descriptorsRed);

  // cout << "number of keypoints blue: " << keypointsBlue.size() << endl;
  // cout << "number of keypoints green: " << keypointsGreen.size() << endl;
  // cout << "number of keypoints red: " << keypointsRed.size() << endl;

  Mat imgBlue;
  drawKeypoints(blue, keypointsBlue, imgBlue, Scalar(255,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  Mat imgGreen;
  drawKeypoints(green, keypointsGreen, imgGreen, Scalar(0,255,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  Mat imgRed;
  drawKeypoints(red, keypointsRed, imgRed, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);



  // Match features.
  ///
  /// YOUR CODE HERE
  ///
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::MatcherType::BRUTEFORCE_HAMMING);

  // Match features between blue and Green channels
  ///
  /// YOUR CODE HERE
  ///
  std::vector<DMatch> matchesBlueGreen;
  matcher->match(descriptorsBlue, descriptorsGreen, matchesBlueGreen);
  // Sort matches by score
  std::sort(matchesBlueGreen.begin(), matchesBlueGreen.end());
  // Remove not so good matches
  int numGoodMatchesBG = matchesBlueGreen.size() * GOOD_MATCH_PERCENT;
  matchesBlueGreen.erase(matchesBlueGreen.begin()+numGoodMatchesBG, matchesBlueGreen.end());
  // Draw top matches
  Mat imMatchesBlueGreen;
  drawMatches(blue, keypointsBlue, green, keypointsGreen, matchesBlueGreen, imMatchesBlueGreen);


  // Match features between Red and Green channels
  ///
  /// YOUR CODE HERE
  ///
  std::vector<DMatch> matchesRedGreen;
  matcher->match(descriptorsRed, descriptorsGreen, matchesRedGreen);
  // Sort matches by score
  std::sort(matchesRedGreen.begin(), matchesRedGreen.end());
  // Remove not so good matches
  int numGoodMatchesRG = matchesRedGreen.size() * GOOD_MATCH_PERCENT;
  matchesRedGreen.erase(matchesRedGreen.begin()+numGoodMatchesRG, matchesRedGreen.end());
  // Draw top matches
  Mat imMatchesRedGreen;
  drawMatches(red, keypointsRed, green, keypointsGreen, matchesRedGreen, imMatchesRedGreen);



  // Extract location of good matches Blue / Green
  ///
  /// YOUR CODE HERE
  ///
  std::vector<Point2f> src_ptsBG;
  std::vector<Point2f> dst_ptsBG;
  for(size_t i=0; i<matchesBlueGreen.size(); i++) {
    src_ptsBG.push_back(keypointsBlue[matchesBlueGreen[i].queryIdx].pt);
    dst_ptsBG.push_back(keypointsGreen[matchesBlueGreen[i].trainIdx].pt);
  }
  // cout << src_ptsBG << endl;
  // Find homography
  ///
  /// YOUR CODE HERE
  ///
  Mat hBlueGreen = findHomography(src_ptsBG, dst_ptsBG, RANSAC, 5.0);


  // Extract location of good matches Red / Green
  ///
  /// YOUR CODE HERE
  ///
  std::vector<Point2f> src_ptsRG;
  std::vector<Point2f> dst_ptsRG;
  for(size_t i=0; i<matchesRedGreen.size(); i++) {
    src_ptsRG.push_back(keypointsRed[matchesRedGreen[i].queryIdx].pt);
    dst_ptsRG.push_back(keypointsGreen[matchesRedGreen[i].trainIdx].pt);
  }
  // Find homography
  ///
  /// YOUR CODE HERE
  ///
  Mat hRedGreen = findHomography(src_ptsRG, dst_ptsRG, RANSAC, 5.0);


  // Use homography to find blueWarped and RedWarped images
  ///
  /// YOUR CODE HERE
  ///
  // Mat blueWarped(blue.clone());
  // Mat redWarped(red.clone());


  // hBlueGreen = Mat::eye(3, 3, CV_32F);
  // hRedGreen = Mat::eye(3, 3, CV_32F);

  Mat blueWarped, redWarped;

  // blueWarped.convertTo(blueWarped, CV_64F);
  // redWarped.convertTo(redWarped, CV_64F);
  // hRedGreen.convertTo(hRedGreen, CV_64F);
  // hBlueGreen.convertTo(hBlueGreen, CV_64F);
  // blue.convertTo(blue, CV_64F);
  // red.convertTo(red, CV_64F);

  // cvtColor(blue, blue, COLOR_GRAY2BGR);
  // cvtColor(blueWarped, blueWarped, COLOR_GRAY2BGR);
  // cvtColor(hBlueGreen, hBlueGreen, COLOR_GRAY2BGR);
  // cvtColor(red, red, COLOR_GRAY2BGR);
  // cvtColor(redWarped, redWarped, COLOR_GRAY2BGR);
  // cvtColor(hRedGreen, hRedGreen, COLOR_GRAY2BGR);

  // cout << blue.type() << endl;
  // cout << blueWarped.type() << endl;
  // cout << hRedGreen.type() << endl;
  // cout << red.type() << endl;
  // cout << redWarped.type() << endl;
  // cout << hBlueGreen.type() << endl;


  warpPerspective(blue, blueWarped, hBlueGreen, Size(width, height));
  warpPerspective(red, redWarped, hRedGreen, Size(width, height));


  Mat colorImage;
  vector<Mat> colorImageChannels{blueWarped, green, redWarped};
  merge(colorImageChannels, colorImage);

  imshow(args.winName, colorImage);

  waitKey(0);

  return EXIT_SUCCESS;
}
