
# cd ..

if [ -z "${buildMaya}" ]; then
  echo "$buildMaya"
fi

rm -r ./build

/usr/bin/cmake \
  -D CMAKE_BUILD_TYPE:STRING=$buildType \
  -D CMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
  -D CMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc \
  -D CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/g++ \
  -S /home/"$(whoami)"/Dropbox/code/darkest \
  -B /home/"$(whoami)"/Dropbox/code/darkest/build \
  -G "Unix Makefiles"

/usr/bin/cmake \
  --build /home/"$(whoami)"/Dropbox/code/darkest/build \
  --config $buildType \
  --target all \
  -j 16 \
  --

# If run is not empty
if [ -n "$run" ]; then
  # buildPath="./build/"
  execPat="./build/""$run"
  $execPat
fi
