# CUDA
#./build/bin/stereo_vision_parallel -k $(pwd)/kitti_mini/ -v={VIDEO_MODE} -s={SUBSAMPLING_MODE} -p=0 -f={SCALE_FACTOR:.2f}

# OMP
#./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=0 -p=0 -f=1

# CPU
#OMP_NUM_THREADS=12 ./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=0 -p=0 -f=1

#make all
# ./profileBuild.sh

#echo "BAYESIAN_DISTANCE_THRESH" > tmp2.txt


#for BAYESIAN_DISTANCE_THRESH in {1..200}
for ((BAYESIAN_DISTANCE_THRESH=10;BAYESIAN_DISTANCE_THRESH<=250;BAYESIAN_DISTANCE_THRESH+=5));
do
	echo "$BAYESIAN_DISTANCE_THRESH" >&2 
	echo "$BAYESIAN_DISTANCE_THRESH" >> tmp2.txt
	echo "CPU" >> tmp2.txt
	#./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp2.txt
	./build/bin/stereo_vision_serial -k $(pwd)/kitti_mini/ -v=1 -t=1 -B=$BAYESIAN_DISTANCE_THRESH | grep bay_err= >> tmp2.txt
	# echo "OMP" >> tmp2.txt
	# #./build/bin/stereo_vision_omp -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp2.txt
	# ./build/bin/stereo_vision_omp -k $(pwd)/kitti_mini/ -v=1 -t=1 -B=$BAYESIAN_DISTANCE_THRESH | grep bay_err= >> tmp2.txt
	# echo "CUDA" >> tmp2.txt
	# #./build/bin/stereo_vision_parallel -k $(pwd)/kitti_mini/ -v=1 -s=$SUBSAMPLING_MODE -p=0 -f=$SCALE_FACTOR | grep AVG_FPS= >> tmp2.txt
	# ./build/bin/stereo_vision_parallel -k $(pwd)/kitti_mini/ -v=1 -t=1 -B=$BAYESIAN_DISTANCE_THRESH | grep bay_err= >> tmp2.txt
done

python3 test.py
python3 test.py > results.csv