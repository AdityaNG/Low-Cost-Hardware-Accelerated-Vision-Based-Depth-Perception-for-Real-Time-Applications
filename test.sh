# CUDA
#./build/bin/stereo_vision_parallel -k $(pwd)/kitti_mini/ -v={VIDEO_MODE} -s={SUBSAMPLING_MODE} -p=0 -f={SCALE_FACTOR:.2f}

# OMP
#./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=0 -p=0 -f=1

# CPU
#OMP_NUM_THREADS=12 ./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=0 -p=0 -f=1

#make all
./profileBuild.sh

echo "SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE" > tmp.txt

for SCALE_FACTOR in $(seq 0.5 .1 3.0)
#for SCALE_FACTOR in $(seq 2.5 .1 3.0)
do
    for SUBSAMPLING_MODE in {0..1}
    do
        echo "$SCALE_FACTOR,$SUBSAMPLING_MODE" >&2 
        echo "$SCALE_FACTOR,$SUBSAMPLING_MODE" >> tmp.txt
        echo "CPU" >> tmp.txt
        ./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp.txt
        echo "OMP" >> tmp.txt
        ./build/bin/stereo_vision_omp -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp.txt
        echo "CUDA" >> tmp.txt
        ./build/bin/stereo_vision_parallel -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp.txt
    done

done

python3 test.py
python3 test.py > results.csv

rm tmp.txt