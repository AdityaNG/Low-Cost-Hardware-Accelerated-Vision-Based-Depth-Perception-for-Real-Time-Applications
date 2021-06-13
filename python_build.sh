make shared_library -j12
cp build/bin/stereo_vision.so stereo_vision/build/bin/stereo_vision.so
python3 -m pip install .
