cmake_minimum_required(VERSION 3.20)


# Set the C++ standard to C++23 or later
cmake_policy(SET CMP0048 NEW)


# Name of this example project
project(oa VERSION 0.1.0 DESCRIPTION "The OA")


set(CMAKE_CXX_STANDARD 23)  # Required on mac
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR})


# Find packages
find_package(OpenCV REQUIRED)

# Add subdirectories recursively
add_subdirectory(cpp/api)


function(add_app base_dir tgt_name)
  set(sources ${ARGN})
  list(TRANSFORM sources PREPEND "${base_dir}/")
  add_executable(${tgt_name} ${sources})
  target_link_libraries(${tgt_name} PRIVATE ${OpenCV_LIBS})
endfunction()


## Create the final executable
# add_executable(oa cpp/apps/main.cpp)

## Link the libraries to the executable
# target_link_libraries(oa ${OpenCV_LIBS})
# include_directories(oa ${OpenCV_INCLUDE_DIRS})
# include_directories(${OpenCV_INCLUDE_DIRS})

# Templates
add_app("cpp/camera" "camera" "main.cpp")
# add_app("cpp/image" "image" "main.cpp")
# add_app("cpp/video" "video" "main.cpp")

# OpenCV University
# add_app("cpp/auto-focus" "auto-focus" "main.cpp")
# add_app("cpp/panorama-stiching" "panorama-stiching" "main.cpp")
# add_app("cpp/morphology" "morphology" "main.cpp")
# add_app("cpp/image-alignment" "image-alignment" "main.cpp")
# add_app("cpp/document-scanner" "document-scanner" "main.cpp")
# add_app("cpp/ig-filters" "ig-filters" "main.cpp")
# add_app("cpp/chroma-keying" "chroma-keying" "main.cpp")
# add_app("cpp/blemish-removal" "blemish-removal" "main.cpp")
# add_app("cpp/dji-fpv" "dji-fpv" "main.cpp")