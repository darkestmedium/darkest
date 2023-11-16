#pragma once

// System Includes
#include <iostream>
#include <sstream>
#include <cmath>
#include <string>
#include <vector>


// Custom




namespace Text {
  /* Text wrapper namespace. */


  inline std::string decimal_precision(const double number, const unsigned int decimal_digits=2) {
    /* Set the decimal precision od doubles.

    Args:
      number (double): Given number

    Returns:
      decimal_digits (unsigned int): Decimal precision

    */
    std::stringstream str_stream;
    str_stream.precision(decimal_digits);  // set # places after decimal
    str_stream << std::fixed;
    str_stream << number;
    return str_stream.str();
  }

}
