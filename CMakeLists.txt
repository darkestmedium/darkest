cmake_minimum_required(VERSION 3.20)


# Set the C++ standard to C++23 or later
cmake_policy(SET CMP0048 NEW)

# Name of this example project
project(oa VERSION 0.1.0 DESCRIPTION "The OA")

set(CMAKE_CXX_STANDARD 23)  # Required on mac
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_MoODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cgcmake/modules)


# Find packages
find_package(OpenCV REQUIRED)

# Add subdirectories recursively
add_subdirectory(cpp/api)


if (${BUILD_MAYA})
  # Build the maya plugin
  add_subdirectory(cpp/maya)
endif()


function(add_module base_dir tgt_name)
  set(sources ${ARGN})
  list(TRANSFORM sources PREPEND "${base_dir}/")
  add_executable(${tgt_name} ${sources})
  target_link_libraries(${tgt_name} PRIVATE ${OpenCV_LIBS})
endfunction()


# # Create the final executable
# add_executable(oa cpp/apps/main.cpp)

# # Link the libraries to the executable
# target_link_libraries(oa ${OpenCV_LIBS})
# include_directories(oa ${OpenCV_INCLUDE_DIRS})
# include_directories(${OpenCV_INCLUDE_DIRS})

## Tutorials
# add_module("cpp/videocapture" "videocapture" "main.cpp")
# add_module("cpp/qrcode" "qrcode" "main.cpp")
add_module("cpp/face-annotation-tool" "face-annotation-tool" "main.cpp")

