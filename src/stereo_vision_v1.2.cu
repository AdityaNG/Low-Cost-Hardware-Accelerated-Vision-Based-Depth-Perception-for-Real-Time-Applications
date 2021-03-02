//#include <curl/curl.h>
#include <exception>
#include <iostream>
#include <opencv4/opencv2/highgui.hpp>
#include <vector>
#include <thread> 
#include <stdlib.h>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string.h>
#include <math.h>
#include <popt.h>
#include <future>
#include <omp.h>

#include "yolo/yolo.hpp"
#include "elas_cuda_openmp/elas.h"
#include "elas_cuda_openmp/elas_gpu.h"
#include "graphing/graphing.h"
#include "cleanup/cleanup.hpp"
#include "bayesian/bayesian.h"


#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

std::vector<OBJ> obj_list, pred_list;

using namespace cv;
using namespace std;

#define shrink_factor 2 // Modify to change the image resize factor

#define start_timer auto start = chrono::high_resolution_clock::now();  
#define end_timer(var)\
  auto end = chrono::high_resolution_clock::now();\
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count();\
  time_taken *= 1e-9;\
  var = time_taken;                                             

//////////////////////////////////////// Globals ///////////////////////////////////////////////////////
Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy;
Mat left_img_OLD, right_img_OLD, dmapOLD;
Vec3d T;
FileStorage calib_file;

Size out_img_size;
Size calib_img_size;
int calib_width = 1242, calib_height = 375,
    out_width = 1242/shrink_factor, out_height = 375/shrink_factor;

const char* kitti_path;
const char* calib_file_name = "calibration/kitti_2011_09_26.yml";

double pc_t = 0, yd_t = 0, t_t = 0; // For calculating timings

int video_mode = 0;
int debug = 0;
int draw_points = 0;
int frame_skip = 1;
int play_video = 0;

// Cuda globals
double *d_XT, *d_XR, *d_Q;
uchar *d_dmap; // Disparity map needs to be pushed to GPU
double3 *points; // Holds the coordinates of each pixel in 3D space
double3 *d_points;
uchar4 *color = NULL;
cudaStream_t s1;
const dim3 blockSize(32, 32, 1);
const dim3 gridSize((out_width / blockSize.x) + 1, (out_height / blockSize.y) + 1, 1);
////////////////////////////////////////////////////////////////////////////////////////////////////////
/*
Elas::parameters param(Elas::MIDDLEBURY);
param.postprocess_only_left = true;
ElasGPU elas(param);*/
  //Elas::parameters param(Elas::ROBOTICS);
  //Elas::parameters param;
  

  //param.postprocess_only_left = false;
  
  //Elas elas(param);
  

void cudaInit(){
  // Cuda Init
  cudaMalloc(&d_XT, sizeof(double) * 3);
  cudaMalloc(&d_XR, sizeof(double) * 9);
  cudaMalloc(&d_Q, sizeof(double) * 16);
  cudaMalloc(&d_dmap, sizeof(uchar) * out_width * out_height);
  cudaMalloc(&d_points, sizeof(double3) * out_width * out_height);
  points = (double3*)malloc(sizeof(double3) * out_width * out_height);
  cudaStreamCreate(&s1);  
  printf("CUDA Init done\n");
}

void clean(){
  // Cuda Cleanup
  printf("Exitting the program.....\n");
  destroyAllWindows();
  free(points);
  cudaStreamDestroy(s1);
  cudaFree(d_XR);
  cudaFree(d_XT);
  cudaFree(d_Q);
  cudaFree(d_points);
  cudaFree(d_dmap);
  //elas.cudaDest();
  exit(0);
}

int constrain(int a, int lb, int ub) {
  if (a<lb)
    return lb;
  else if (a>ub)
    return ub;
  else
    return a;
}

/*
 * Function:  composeRotationCamToRobot 
 * --------------------
 * Given a (x,y,z) rotation params, a corresponding 3D rotation matrix is generated
 *
 *  float x: The x rotation
 *  float y: The y rotation
 *  float z: The z rotation
 *  returns: Mat The 3D rotation matrix
 *
 */
Mat composeRotationCamToRobot(float x, float y, float z) {
  Mat X = Mat::eye(3, 3, CV_64FC1);
  Mat Y = Mat::eye(3, 3, CV_64FC1);
  Mat Z = Mat::eye(3, 3, CV_64FC1);
  
  X.at<double>(1,1) = cos(x);
  X.at<double>(1,2) = -sin(x);
  X.at<double>(2,1) = sin(x);
  X.at<double>(2,2) = cos(x);

  Y.at<double>(0,0) = cos(y);
  Y.at<double>(0,2) = sin(y);
  Y.at<double>(2,0) = -sin(y);
  Y.at<double>(2,2) = cos(y);

  Z.at<double>(0,0) = cos(z);
  Z.at<double>(0,1) = -sin(z);
  Z.at<double>(1,0) = sin(z);
  Z.at<double>(1,1) = cos(z);
  
  return Z*Y*X;
}

/*
 * Function:  composeTranslationCamToRobot 
 * --------------------
 * Given a (x,y,z) translation params, a corresponding 3D tranlation matrix is generated
 *
 *  float x: The x translation
 *  float y: The y translation
 *  float z: The z translation
 *  returns: Mat The 3D tranlation matrix
 *
 */
Mat composeTranslationCamToRobot(float x, float y, float z) {
  return (Mat_<double>(3,1) << x, y, z);
}

/*
 * Function:  publishPointCloud 
 * --------------------
 * Given a disparity map, a corresponding 3D point cloud can be easily constructed. 
 * The Q matrix stored in the calibration file is used for this conversion. 
 * The reconstruction is mathematically expressed by the following matrix equation.
 *
 *               [  [1 0 0      -Cx         ];
 * (X,Y,Z,W)^T =    [0 1 0      -Cy         ];     . [x y d(x,y) 1]^T
 *                  [0 0 0      f           ]; 
 *                  [0 0 -1/Tx  (Cx-C'x)/Tx ]; ] 
 *
 * d(x,y)  is the disparity of a point (x,y) in the left image
 * The 4X4 matrix dentoes the Q matrix 
 *
 * The point cloud generated is in the reference frame of the left camera. 
 * Hence a transformation (XR, XT) is applied to transform the point cloud into a different reference frame 
 * (as required by the user). The transformation equation is as follows
 * PB = R × PA + T
 *
 * Q Matrix
 * [1, 0, 0,                  -339.7460250854492;
 *  0, 1, 0,                  -110.0997492116292;
 *  0, 0, 0,                  455.4106857822576;
 *  0, 0, 1.861616069957151,  -0]
 *
 *  Mat& img_left: The input left image - set of points (x, y)
 *  Mat& dmap: input disparity map d(x, y)
 *  returns: void
 *
 */


 __global__ void parallel(const uchar *dmap, double3 *points, int rows, int cols, const double *d_XT, const double *d_XR, const double *d_Q){
  // Calculating the coordinates of the pixel
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	// To prevent trying to access data outside the image
	if (x >= cols || y >= rows)
        return;
    
  int pixelPosition = y * cols + x;
  uchar d = dmap[pixelPosition];
  //if(d < 2) return;

  double pos[4];
  for(int j = 0; j<4; j++) pos[j] = d_Q[4*j + 0]*x + d_Q[4*j + 1]*y + d_Q[4*j + 2]*d + d_Q[4*j + 3];
    
  double X = pos[0] / pos[3];
  double Y = pos[1] / pos[3];
  double Z = pos[2] / pos[3];

  double point[3];
  for(int j = 0; j<3; j++) point[j] = d_XR[3*j + 0]*X + d_XR[3*j + 1]*Y + d_XR[3*j + 2]*Z + d_XT[j];
    
  points[pixelPosition] = make_double3(point[0], point[1], point[2]);
}

 void publishPointCloud(Mat& img_left, Mat& dmap) { 
  if (img_left.empty() || dmap.empty()) {
    printf("(empty)\t");
    return;
  }
  if (debug == 1) {
    XR = composeRotationCamToRobot(1.3 ,-3.14,1.57);
    XT = composeTranslationCamToRobot(0.0,0.0,0.28);
    cout << "Rotation matrix: " << XR << endl;
    cout << "Translation matrix: " << XT << endl;
  }

  start_timer; 
  
  cudaMemcpyAsync(d_dmap, dmap.data, sizeof(uchar) * out_width * out_height, cudaMemcpyHostToDevice, s1);
  cudaMemcpy(d_XT, XT.data, sizeof(double) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_XR, XR.data, sizeof(double) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q, Q.data, sizeof(double) * 16, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  parallel <<<gridSize, blockSize, 0, s1>>> (d_dmap, d_points, out_height, out_width, d_XT, d_XR, d_Q);

  cudaDeviceSynchronize();
  cudaMemcpy(points, d_points, sizeof(double3) * out_width * out_height, cudaMemcpyDeviceToHost);

  for(auto& object : obj_list) {
    /*
    int i_lb = constrain(object.x + object.w/2, 0, img_left.cols-1), 
    i_ub = i_lb + 1, 
    j_lb = constrain(object.y + object.h/2, 0, img_left.rows-1), 
    j_ub = j_lb + 1;
    */
    int i_lb = constrain(object.x, 0, img_left.cols-1), 
    i_ub = constrain(object.x + object.w, 0, img_left.cols-1), 
    j_lb = constrain(object.y, 0, img_left.rows-1), 
    j_ub = constrain(object.y + object.h, 0, img_left.rows-1);
    double X=0, Y=0, Z=0;

    #pragma omp parallel for //reduction(+:X,Y,Z)
    for(int i = i_lb; i < i_ub; i++) {
      for(int j = j_lb; j < j_ub; j++) {   
        X += points[j*out_width + i].x;  
        Y += points[j*out_width + i].y;     
        Z += points[j*out_width + i].z;  
      }
    } 
    //appendOBJECTS(X/((i_ub-i_lb)*(j_ub-j_lb)), Y/((i_ub-i_lb)*(j_ub-j_lb)), Z/((i_ub-i_lb)*(j_ub-j_lb)), object.r, object.g, object.b); 
    //if (draw_points == 1)
    appendOBJECTS(Y/((i_ub-i_lb)*(j_ub-j_lb)), -Z/((i_ub-i_lb)*(j_ub-j_lb)), X/((i_ub-i_lb)*(j_ub-j_lb)), object.r, object.g, object.b); 
  }
   
  if (!dmap.empty()) {
    // TODO : Do something
  }
  //updateGraph();
  end_timer(pc_t);
}

/*
 * Function:  generateDisparityMap 
 * --------------------
 * This function computes the dense disparity map using LIBELAS, and returns an 8-bit grayscale image Mat.
 * The disparity map is constructed with the left image as reference. The parameters for LIBELAS can be changed in the file src/elas/elas.h.
 * Any method other than LIBELAS can be implemented inside the generateDisparityMap function to generate disparity maps. One can use OpenCV’s StereoBM class as well. The output should be a 8-bit grayscale image.
 *
 *  Mat& left: The input left image
 *  Mat& right: The input right image
 *  returns: Mat output 8-bit grayscale image
 *
 */
Mat generateDisparityMap(Mat& left, Mat& right) {
  resetOBJECTS();
  if (left.empty() || right.empty()) 
    return left;
  const Size imsize = left.size();
  const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  
  static Elas::parameters param(Elas::MIDDLEBURY);
  //Elas::parameters param(Elas::ROBOTICS);
  //Elas::parameters param;
  
  param.postprocess_only_left = true;
  //param.postprocess_only_left = false;
  
  //Elas elas(param);
  static ElasGPU elas(param);
  
  elas.process(left.data, right.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
  Mat dmap = Mat(out_img_size, CV_8UC1, Scalar(0));
  
  leftdpf.convertTo(dmap, CV_8UC1, 4.0);
  
  return dmap;
}

/*
 * Function:  imgCallback 
 * --------------------
 * Loads the input images into Mats
 * Undistorts and Rectifies the images with remap()
 * Generates disparity map with generateDisparityMap(img_left, img_right)
 * Displays output with imshow() and publishPointCloud()
 *
 *  const char* left_img_topic: path to left image
 *  const char* right_img_topic: path to right image
 *  returns: void
 *
 */

void imgCallback_video() {
  Mat left_img = left_img_OLD; Mat right_img = right_img_OLD;
  if (left_img.empty() || right_img.empty()){
    //printf("%s\n",left_img_topic);
    return;
  }

  Mat img_left, img_right, img_left_color_flip;

  cvtColor(left_img, img_left, COLOR_BGRA2GRAY);
  cvtColor(right_img, img_right, COLOR_BGRA2GRAY);

  //remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  start_timer;   
  dmapOLD = generateDisparityMap(img_left, img_right);  
  end_timer(yd_t);
}

void imgCallback(const char* left_img_topic, const char* right_img_topic, int wait=0) {
printf("imgCalback called\n");
  Mat tmpL_Color = imread(left_img_topic, IMREAD_UNCHANGED);
  Mat tmpL = imread(left_img_topic, IMREAD_GRAYSCALE);
  Mat tmpR = imread(right_img_topic, IMREAD_GRAYSCALE);
  
  if (tmpL.empty() || tmpR.empty()) return;

  resize(tmpL_Color, tmpL_Color, out_img_size);
  resize(tmpL, tmpL, out_img_size);
  resize(tmpR, tmpR, out_img_size);

  Mat frame = tmpL_Color.clone();
  Mat img_left, img_right, img_left_color, img_left_color_flip;  
  
  img_left = tmpL; img_right = tmpR;

  //remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  start_timer;
  auto f = std::async(std::launch::async, processYOLO, tmpL_Color); // Asynchronous call to YOLO
  Mat dmap = generateDisparityMap(img_left, img_right);  
  obj_list = f.get(); // Getting obj_list from the future object which the async call returns to f  
  end_timer(yd_t);

  publishPointCloud(frame, dmap);
  
  flip(tmpL_Color, img_left_color_flip,1);
  
  imshow("LEFT_C", img_left_color_flip);
}

/*
 * Function:  findRectificationMap 
 * --------------------
 * This function computes all the projection matrices and 
 * the rectification transformations using the stereoRectify 
 * and initUndistortRectifyMap functions respectively.
 *
 *  FileStorage& calib_file: The List in question
 *  Size finalSize: The data to tbe inserted
 *  returns: void
 *
 */
void findRectificationMap(FileStorage& calib_file, Size finalSize) {
  Rect validRoi[2];
  cout << "Starting rectification" << endl;

  /*
  void cv::stereoRectify  ( 
    InputArray  cameraMatrix1,
    InputArray  distCoeffs1,
    InputArray  cameraMatrix2,
    InputArray  distCoeffs2,
    Size        imageSize,
    InputArray  R,
    InputArray  T,
    OutputArray R1,
    OutputArray R2,
    OutputArray P1,
    OutputArray P2,
    OutputArray Q,
    int         flags = CALIB_ZERO_DISPARITY,
    double      alpha = -1,
    Size        newImageSize = Size(),
    Rect *      validPixROI1 = 0,
    Rect *      validPixROI2 = 0 
  )

  stereoRectify
  Computes rectification transforms for each head of a calibrated stereo camera.

  Paramers 
    cameraMatrix1   First camera intrinsic matrix.
    distCoeffs1     First camera distortion parameters.
    cameraMatrix2   Second camera intrinsic matrix.
    distCoeffs2     Second camera distortion parameters.
    imageSize       Size of the image used for stereo calibration.
    R               Rotation matrix from the coordinate system of the first camera to the second camera, see stereoCalibrate.
    T               Translation vector from the coordinate system of the first camera to the second camera, see stereoCalibrate.
    R1  Output 3x3  rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
    R2  Output 3x3  rectification transform (rotation matrix) for the second camera. This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system.
    P1  Output 3x4  projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
    P2  Output 3x4  projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
    Q   Output 4×4  disparity-to-depth mapping matrix (see reprojectImageTo3D).
    flags           Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
    alpha           Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.
    newImageSize    New image resolution after rectification. The same size should be passed to initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original imageSize . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
    validPixROI1    Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
    validPixROI2    Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
  */
  //stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, 
                CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);
  
  //P1 = (Mat_<double>(3,4) << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03);
  //P2 = (Mat_<double>(3,4) << 7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02, 0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.199936000000e+00, 0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.729905000000e-03);

  /*
  void cv::initUndistortRectifyMap  ( 
    InputArray  cameraMatrix,
    InputArray  distCoeffs,
    InputArray  R,
    InputArray  newCameraMatrix,
    Size  size,
    int   m1type,
    OutputArray   map1,
    OutputArray   map2 
  )

  initUndistortRectifyMap
    Computes the undistortion and rectification transformation map.
    The function computes the joint undistortion and rectification transformation and represents the result in the form of maps for remap. 
    The undistorted image looks like original, as if it is captured with a camera using the camera matrix =newCameraMatrix and zero distortion. 
    In case of a monocular camera, newCameraMatrix is usually equal to cameraMatrix, or it can be computed by getOptimalNewCameraMatrix for a better control over scaling. 
    In case of a stereo camera, newCameraMatrix is normally set to P1 or P2 computed by stereoRectify .
    Also, this new camera is oriented differently in the coordinate space, according to R. 
    That, for example, helps to align two heads of a stereo camera so that the epipolar lines on both images become horizontal and have the same y- coordinate (in case of a horizontally aligned stereo camera).

  Paramers 
    cameraMatrix    Input camera matrix A=[fx 0 cx; 0 fy cy; 0 0 1].
    distCoeffs      Input vector of distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,τx,τy]]]]) of 4, 5, 8, 12 or 14 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
    R Optional      rectification transformation in the object space (3x3 matrix). R1 or R2 , computed by stereoRectify can be passed here. If the matrix is empty, the identity transformation is assumed. In cvInitUndistortMap R assumed to be an identity matrix.
    newCameraMatrix New camera matrix A′=[f'x 0 c'x; 0 f'y c'y; 0 0 1].
    size            Undistorted image size.
    m1type          Type of the first output map that can be CV_32FC1, CV_32FC2 or CV_16SC2, see convertMaps
    map1            The first output map.
    map2            The second output map.
  */
  cv::initUndistortRectifyMap(K1, D1, R1, P1, finalSize, CV_32F, lmapx, lmapy);
  cv::initUndistortRectifyMap(K2, D2, R2, P2, finalSize, CV_32F, rmapx, rmapy);
  
  cout << "------------------" << endl;
  cout << "Done rectification" << endl;
  
}

void next(){
  static int iImage=0;
  if (video_mode){
    char left_img_topic[128], right_img_topic[128];
    size_t max_files = 465; // Just hardcoded the value for now

    Mat left_img, right_img, dmap, YOLOL_Color, img_left_color_flip;
    //thread th1(imgCallback_video);
    thread th1;

    play_video = 1;
    while (play_video){
      for(int iFrame = 0; iFrame < max_files; iFrame++){
        if (t_t!=0) printf("(FPS=%f) ", 1/t_t);
        
        start_timer;        
        strcpy(left_img_topic , format("%s/video/testing/image_02/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    
        strcpy(right_img_topic, format("%s/video/testing/image_03/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    

        left_img = imread(left_img_topic, IMREAD_UNCHANGED);
        right_img = imread(right_img_topic, IMREAD_UNCHANGED);
        resize(left_img, left_img, out_img_size);
        resize(right_img, right_img, out_img_size);        

        YOLOL_Color = left_img.clone();
        obj_list = processYOLO(YOLOL_Color);
        pred_list = get_predicted_boxes();
        append_old_objs(obj_list);
        obj_list.insert( obj_list.end(), pred_list.begin(), pred_list.end() );
        
        //auto f = std::async(std::launch::async, processYOLO, YOLOL_Color); // Asynchronous call to YOLO 

        if (iFrame%frame_skip == 0) {
          //printf("(DISP) \t ");
          //imgCallback_video(left_img, right_img, dmap);
          left_img_OLD = left_img.clone();
          right_img_OLD = right_img.clone();

          //disp_parallel = std::async(imgCallback_video);
          th1 = thread(imgCallback_video);
        }
          
        if (iFrame%frame_skip == frame_skip-1) {
          //printf("(JOIN) \t");
          th1.join();
          dmap = dmapOLD.clone();
        }

        printf("(%d, %d) ", dmap.rows, dmap.cols);
        
        //if (iFrame%frame_skip == frame_skip-1) {
        // th1.join();
        //  dmap = dmapOLD.clone();
        //}
    
        Mat rgba;
        cvtColor(left_img, rgba, cv::COLOR_BGR2BGRA);
        color = (uchar4*)rgba.ptr<unsigned char>(0);
        //obj_list = f.get(); // Getting obj_list from the future object which the async call return to f
        publishPointCloud(left_img, dmap);
        //printf("(PC Done) ");
        updateGraph();

        if (0){
          flip(left_img, img_left_color_flip,1);
          namedWindow("Detections", cv::WINDOW_NORMAL); // Needed to allow resizing of the image shown
          namedWindow("Disparity", cv::WINDOW_NORMAL); // Needed to allow resizing of the image shown
          imshow("Detections", YOLOL_Color);
          imshow("Disparity", dmap);
          waitKey(1);
        }
        end_timer(t_t);        
        printf("(t_t=%f, \t yd_t=%f, \t pc_t=%f)\n",t_t, yd_t, pc_t);
      }
    }
  } 
  else {
    printf("Next image\n");
    char left_img_topic[128], right_img_topic[128];
    strcpy(left_img_topic , format("%s/video/testing/image_2/0000/%06d.png", kitti_path, iImage).c_str());    
    strcpy(right_img_topic, format("%s/video/testing/image_3/0000/%06d.png", kitti_path, iImage).c_str());       
    imgCallback(left_img_topic, right_img_topic);
    iImage++;
  }
}

void next_video() {
  play_video = 0;
}

void imageLoop() {
  while (1) next();
}

int main(int argc, const char** argv){  
  initYOLO();

  static struct poptOption options[] = { 
    { "kitti_path",'k',POPT_ARG_STRING,&kitti_path,0,"Path to KITTI Dataset","STR" },
    { "video_mode",'v',POPT_ARG_INT,&video_mode,0,"Set v=1 Kitti video mode","NUM" },
    { "draw_points",'p',POPT_ARG_INT,&draw_points,0,"Set p=1 to plot out points","NUM" },
    { "frame_skip",'f',POPT_ARG_INT,&frame_skip,0,"Set frame_skip to skip disparity generation for f frames","NUM" },
    { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  poptContext poptCONT = poptGetContext("main", argc, argv, options, POPT_CONTEXT_KEEP_FIRST);
  int c = 0; while(c >= 0) c = poptGetNextOpt(poptCONT);

  printf("KITTI Path: %s \n", kitti_path);
  
  calib_img_size = Size(calib_width, calib_height);
  out_img_size = Size(out_width, out_height);
  
  calib_file = FileStorage(calib_file_name, FileStorage::READ);
  calib_file["K1"] >> K1;
  calib_file["K2"] >> K2;
  calib_file["D1"] >> D1;
  calib_file["D2"] >> D2;
  calib_file["R"]  >> R;
  calib_file["T"]  >> T;
  calib_file["XR"] >> XR;
  calib_file["XT"] >> XT;

 
  cout << " K1 : " << K1 << "\n D1 : " << D1 << "\n R1 : " << R1 << "\n P1 : " << P1  
       << "\n K2 : " << K2 << "\n D2 : " << D2 << "\n R2 : " << R2 << "\n P2 : " << P2 << '\n';
  
  findRectificationMap(calib_file, out_img_size);
  
  cudaInit();

  setCallback(next_video);
  thread th1(imageLoop);
  startGraphics(out_width, out_height);
  th1.join();
  clean();
  return 0;
}
