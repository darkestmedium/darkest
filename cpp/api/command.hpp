#pragma once

// System includes
#include <iostream>

// Custom includes
#include "argparse.hpp"



class Command {
  /* Command template class.
  */
public:

  struct Syntax : public argparse::Args {
    std::string &filePath = kwarg("fp,filePath", "Path to the file.").set_default("/home/ccpcpp/Dropbox/code/darkest/resources/images/blemish.png");
    std::string &winName  = kwarg("wn,winName", "Name of the opencv window.").set_default("OpenCV - GTK - Window");
    bool &verbose         = flag("v,verbose", "Toggle verbose");
    bool &help            = flag("h,help", "Display help.");
  };


  virtual void doIt() {};
  virtual void displayHelp() {};

};
