# CUDA
#./build/bin/stereo_vision_parallel -k /home/aditya/Pictures/kitti_mini/ -v={VIDEO_MODE} -s={SUBSAMPLING_MODE} -p=0 -f={SCALE_FACTOR:.2f}

# OMP
#./build/bin/stereo_vision_serial -k /home/aditya/Pictures/kitti_mini/ -v=1 -s=0 -p=0 -f=1

# CPU
#OMP_NUM_THREADS=12 ./build/bin/stereo_vision_serial -k /home/aditya/Pictures/kitti_mini/ -v=1 -s=0 -p=0 -f=1

echo "SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE"
for SCALE_FACTOR in $(seq 0.5 .1 3.0)
do
    for SUBSAMPLING_MODE in {0..1}
    do
        for VIDEO_MODE in {0..1}
        do
            echo "$SCALE_FACTOR,$SUBSAMPLING_MODE,$VIDEO_MODE" >&2 
            echo "$SCALE_FACTOR,$SUBSAMPLING_MODE,$VIDEO_MODE"
            echo "CPU"
            OMP_NUM_THREADS=1 ./build/bin/stereo_vision_serial -k /home/aditya/Pictures/kitti_mini/ -v=$VIDEO_MODE -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS=
            echo "OMP"
            OMP_NUM_THREADS=12 ./build/bin/stereo_vision_serial -k /home/aditya/Pictures/kitti_mini/ -v=$VIDEO_MODE -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS=
            echo "CUDA"
            ./build/bin/stereo_vision_parallel -k /home/aditya/Pictures/kitti_mini/ -v=$VIDEO_MODE -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS=
        done
    done

done


