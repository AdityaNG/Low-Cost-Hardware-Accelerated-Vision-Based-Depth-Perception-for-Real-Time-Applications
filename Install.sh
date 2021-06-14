make shared_library $1 -j12
cp build/bin/stereo_vision.so stereo_vision/bin/stereo_vision.so
python3 -m pip install .
