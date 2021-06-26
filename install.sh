make shared_library serial=1 -j12
make shared_library $1 -j12
rm stereo_vision/bin/stereo_vision.so
cp build/bin/stereo_vision_parallel.so stereo_vision/bin/stereo_vision_parallel.so
cp build/bin/stereo_vision_serial.so stereo_vision/bin/stereo_vision_serial.so
python3 -m pip install .
