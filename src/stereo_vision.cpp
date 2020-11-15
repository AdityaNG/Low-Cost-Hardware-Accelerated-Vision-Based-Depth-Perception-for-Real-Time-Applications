//#include <curl/curl.h>
#include <iostream>
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
#include <experimental/filesystem>  

#include "yolo/yolo.hpp"
#include "elas/elas.h"
#include "graphing/graphing.h"


#define GL_GLEXT_PROTOTYPES
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

std::vector<OBJ> obj_list;

using namespace cv;
using namespace std;
namespace fs = std::experimental::filesystem;


void print_OBJ(OBJ o) {
  cout <<"Name : "<< o.name <<endl;
  cout <<"\t x : "<< o.x << '\n';
  cout <<"\t y : "<< o.y << '\n';
  cout <<"\t h : "<< o.h << '\n';
  cout <<"\t w : "<< o.w << '\n';
  cout <<"\t c : "<< o.c << '\n';
  cout <<"--------"<<endl;
}

int constrain(int a, int lb, int ub) {
  if (a<lb)
    return lb;
  else if (a>ub)
    return ub;
  else
    return a;
}

Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy;
Vec3d T;
FileStorage calib_file;

Size out_img_size;
Size calib_img_size;
const char* kitti_path;
int video_mode = 0;
int debug = 0;
int draw_points = 0;
int frame_skip = 1;

double pc_t = 0, yd_t = 0, t_t = 0;

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
void publishPointCloud(Mat& img_left, Mat& dmap) {

  if (img_left.empty() || dmap.empty()) {
    return;
  }

  auto start = chrono::high_resolution_clock::now();   
  // unsync the I/O of C and C++. 
  ios_base::sync_with_stdio(false);
  if (debug == 1) {
    //XR = composeRotationCamToRobot(1.3 ,-3.14,1.57);
    XR = composeRotationCamToRobot(M_PI/3, 0, 0); //M_PI
    XT = composeTranslationCamToRobot(-4,-1.0,1.7);
    cout << "Rotation matrix: " << XR << endl;
    cout << "Translation matrix: " << XT << endl;
  }
  Mat V = Mat(4, 1, CV_64FC1);
  Mat pos = Mat(4, 1, CV_64FC1);
  vector< Point3d > points;

  //cout << "Q matrix: " << Q << endl;

  //cout<<"img.size : "<<img_left.cols<<", "<<img_left.rows<<endl;

  //cout << "POINTS"<<endl;
  int BOUNDARY = 50;
  if (draw_points) {
  for (int i = 0; i < img_left.cols; i++) {
    for (int j = 0; j < img_left.rows; j++) {
      int d = dmap.at<uchar>(j,i);
      //cout<<d<<endl;
      // if low disparity, then ignore
      if (d < 2) {
        continue;
      }
      // V is the vector to be multiplied to Q to get
      // the 3D homogenous coordinates of the image point
      V.at<double>(0,0) = (double)(i);
      V.at<double>(1,0) = (double)(j);
      V.at<double>(2,0) = (double)d;
      V.at<double>(3,0) = 1.;
      
      pos = Q * V; // 3D homogeneous coordinate
      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
      Mat point3d_cam = Mat(3, 1, CV_64FC1);
      point3d_cam.at<double>(0,0) = X;
      point3d_cam.at<double>(1,0) = Y;
      point3d_cam.at<double>(2,0) = Z;
      // transform 3D point from camera frame to robot frame
      Mat point3d_robot = XR * point3d_cam + XT;
      points.push_back(Point3d(point3d_robot));
      
      int32_t red, blue, green;
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];
      int32_t rgb = (red << 16 | green << 8 | blue);
      //ch.values.push_back(*reinterpret_cast<float*>(&rgb));

      appendPOINT(X, Y, Z, red/255.0, green/255.0, blue/255.0);

      //cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
    }
  }
  }
  
  //cout << "pc : " << fixed << time_taken << setprecision(9); 
  //cout << " \t";
  //cout<<"POINTS_END\nBOXES\n";

  for (auto& object : obj_list) {
    //print_OBJ(object);
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

    for (int i = i_lb; i < i_ub; i++) {
      for (int j = j_lb; j < j_ub; j++) {
        int d = dmap.at<uchar>(j,i);
        
        if (d < 2) {
          continue;
        }
        
        V.at<double>(0,0) = (double)(i);
        V.at<double>(1,0) = (double)(j);
        V.at<double>(2,0) = (double)d;
        V.at<double>(3,0) = 1.;
        
        pos = Q * V; // 3D homogeneous coordinate
        X += pos.at<double>(0,0) / pos.at<double>(3,0);
        Y += pos.at<double>(1,0) / pos.at<double>(3,0);
        Z += pos.at<double>(2,0) / pos.at<double>(3,0);
        Mat point3d_cam = Mat(3, 1, CV_64FC1);
        point3d_cam.at<double>(0,0) = X;
        point3d_cam.at<double>(1,0) = Y;
        point3d_cam.at<double>(2,0) = Z;
        // transform 3D point from camera frame to robot frame
        Mat point3d_robot = XR * point3d_cam + XT;
        points.push_back(Point3d(point3d_robot));
        
        int32_t red, blue, green;
        red = img_left.at<Vec3b>(j,i)[2];
        green = img_left.at<Vec3b>(j,i)[1];
        blue = img_left.at<Vec3b>(j,i)[0];
        int32_t rgb = (red << 16 | green << 8 | blue);
        
      }
    }
    appendOBJECTS(X/((i_ub-i_lb)*(j_ub-j_lb)), Y/((i_ub-i_lb)*(j_ub-j_lb)), Z/((i_ub-i_lb)*(j_ub-j_lb)), object.r, object.g, object.b); 
  }
  //cout<<"BOXES_END\n";

  //system ("touch ../plotter/3D_maps/reload_check");
  
  if (!dmap.empty()) {
    // TODO : Do something
  }
  //updateGraph();
  auto end = chrono::high_resolution_clock::now();   
  // Calculating total time taken by the program. 
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
  time_taken *= 1e-9;   
  pc_t = time_taken;

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
  resetPOINTS();
  resetOBJECTS();
  if (left.empty() || right.empty()) 
    return left;
  const Size imsize = left.size();
  const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  Mat leftdpf = Mat::zeros(imsize, CV_32F);
  Mat rightdpf = Mat::zeros(imsize, CV_32F);

  Elas::parameters param(Elas::MIDDLEBURY);
  //Elas::parameters param(Elas::ROBOTICS);
  //Elas::parameters param;
  
  param.postprocess_only_left = true;
  //param.postprocess_only_left = false;
  
  Elas elas(param);
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
Mat left_img_OLD, right_img_OLD, dmapOLD;
void imgCallback_video() {
  Mat left_img = left_img_OLD; Mat& right_img = right_img_OLD; Mat& dmap = dmapOLD;
  if (left_img.empty() || right_img.empty()){
    //printf("%s\n",left_img_topic);
    return;
  }

  Mat img_left, img_right, img_left_color_flip;

  cvtColor(left_img, img_left, COLOR_BGRA2GRAY);
  cvtColor(right_img, img_right, COLOR_BGRA2GRAY);

  //remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  auto start = chrono::high_resolution_clock::now();   
  ios_base::sync_with_stdio(false);

  
  dmap = generateDisparityMap(img_left, img_right);
  
  
  auto end = chrono::high_resolution_clock::now();   
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
  time_taken *= 1e-9;   
  yd_t = time_taken;
}

void imgCallback(const char* left_img_topic, const char* right_img_topic, int wait=0) {
  Mat tmpL_Color = imread(left_img_topic, IMREAD_UNCHANGED);
  Mat tmpL = imread(left_img_topic, IMREAD_GRAYSCALE);
  Mat tmpR = imread(right_img_topic, IMREAD_GRAYSCALE);
  if (tmpL.empty() || tmpR.empty()){
    //printf("%s\n",left_img_topic);
    return;
  }

  resize(tmpL_Color, tmpL_Color, out_img_size);
  resize(tmpL, tmpL, out_img_size);
  resize(tmpR, tmpR, out_img_size);

  cv::Mat frame = tmpL_Color.clone();
  
  Mat img_left, img_right, img_left_color, img_left_color_flip;
  
  
  img_left = tmpL; img_right = tmpR;

  //remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  auto f = std::async(std::launch::async, processYOLO, tmpL_Color); // Asynchronous call to YOLO
  
  auto start = chrono::high_resolution_clock::now();   
  ios_base::sync_with_stdio(false);

  //obj_list = processYOLO(tmpL_Color);

  Mat dmap = generateDisparityMap(img_left, img_right);
  
  
  auto end = chrono::high_resolution_clock::now();   
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
  time_taken *= 1e-9;   
  yd_t = time_taken;

  obj_list.empty();
  obj_list = f.get(); // Getting obj_list from the future object which the async call return to f

  //cout << "; Y+D " << fixed << time_taken << setprecision(9); 
  //cout << "\t";

  publishPointCloud(frame, dmap);
  
  flip(tmpL_Color, img_left_color_flip,1);
  
  imshow("LEFT_C", img_left_color_flip);
  //imshow("DISP", dmap);
  //waitKey(2000);
  waitKey(wait);
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

std::size_t number_of_files_in_directory(fs::path path)
{
    using fs::directory_iterator;
    using fp = bool (*)( const fs::path&);
    return std::count_if(directory_iterator(path), directory_iterator{}, (fp)fs::is_regular_file);
}

const char* calib_file_name = "calibration/kitti_2011_09_26.yml";
int calib_width, calib_height, out_width, out_height;
int play_video = 0;

void next() {
  static int iImage=0;
  if (video_mode) {
    char left_img_topic[128];
    char right_img_topic[128];
    char right_img_dir[128];
    std::strcpy(right_img_dir , format("%svideo/testing/image_03/%04d/", kitti_path, iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";
    fs::path path_to_folder(right_img_dir);
    size_t max_files = number_of_files_in_directory(path_to_folder);

    Mat left_img, right_img, dmap, YOLOL_Color, img_left_color_flip;
    //thread th1(imgCallback_video);
    thread th1;

    play_video = 1;
    while (play_video)
    {
      for (int iFrame = 0; iFrame < max_files; iFrame++)
      {
        auto start = chrono::high_resolution_clock::now();   
        ios_base::sync_with_stdio(false);
        
        std::strcpy(left_img_topic  , format("%s/video/testing/image_02/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
        std::strcpy(right_img_topic , format("%s/video/testing/image_03/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";

        left_img = imread(left_img_topic, IMREAD_UNCHANGED);
        right_img = imread(right_img_topic, IMREAD_UNCHANGED);
        resize(left_img, left_img, out_img_size);
        resize(right_img, right_img, out_img_size);

        YOLOL_Color = left_img.clone();

        obj_list = processYOLO(YOLOL_Color);
        //auto f = std::async(std::launch::async, processYOLO, YOLOL_Color); // Asynchronous call to YOLO

        if ( iFrame%frame_skip == 0 ) {
          //imgCallback_video(left_img, right_img, dmap);
          left_img_OLD = left_img.clone();
          right_img_OLD = right_img.clone();

          //disp_parallel = std::async(imgCallback_video);
          th1 = thread(imgCallback_video);
        }
          
        if (iFrame%frame_skip == frame_skip-1) {
          th1.join();
          dmap = dmapOLD.clone();
        }
    

        //obj_list = f.get(); // Getting obj_list from the future object which the async call return to f
        publishPointCloud(left_img, dmap);
        updateGraph();

        if (1) {
          //flip(YOLOL_Color, img_left_color_flip,1);
          flip(left_img, img_left_color_flip,1);
          
          imshow("LEFT_C", img_left_color_flip);
          //imshow("DISP", dmap);
          
          waitKey(1);
        }

        auto end = chrono::high_resolution_clock::now();   
        // Calculating total time taken by the program. 
        double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
        time_taken *= 1e-9;   
        t_t = time_taken;
        
        printf("(t_t=%f, \t yd_t=%f, \t pc_t=%f)\n",t_t, yd_t, pc_t);
      }
    }
  } else {
    printf("Next image\n");

    char left_img_topic[128];
    char right_img_topic[128];

    std::strcpy(left_img_topic  , format("%s/object/testing/image_2/%06d.png", kitti_path,  iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
    std::strcpy(right_img_topic , format("%s/object/testing/image_3/%06d.png", kitti_path, iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";
  
    
    imgCallback(left_img_topic, right_img_topic);
    iImage++;
  }
}

void next_video() {
  play_video = 0;
}

void imageLoop() {
  while (1)
  {
    next();
  }
}

int main(int argc, const char** argv) {
  
  initYOLO();

  static struct poptOption options[] = { //
    { "kitti_path",'k',POPT_ARG_STRING,&kitti_path,0,"Path to KITTI Dataset","STR" },
    { "video_mode",'v',POPT_ARG_INT,&video_mode,0,"Set v=1 Kitti video mode","NUM" },
    { "draw_points",'p',POPT_ARG_INT,&draw_points,0,"Set p=1 to plot out points","NUM" },
    { "frame_skip",'f',POPT_ARG_INT,&frame_skip,0,"Set frame_skip to skip disparity generation for f frames","NUM" },
    { "debug",'d',POPT_ARG_INT,&debug,0,"Set d=1 for cam to robot frame calibration","NUM" },
    POPT_AUTOHELP
    { NULL, 0, 0, NULL, 0, NULL, NULL }
  };

  poptContext poptCONT = poptGetContext("main", argc, argv, options, POPT_CONTEXT_KEEP_FIRST);
  int c; while((c = poptGetNextOpt(poptCONT)) >= 0) {}

  printf("KITTI Path: %s \n", kitti_path);

  calib_width = 1242;
  calib_height = 375;
  out_width = 1242/4;
  out_height = 375/4;
  
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

 
  cout << " K1 : " << "D1 : " << "R1 : " << "P1 : " << "K2 : " << "D2 : " << "R2 : " << "P2 : " << endl;
  cout <<  K1 << endl << D1 << endl << R1 << endl << P1 << endl << K2 << endl << D2 << endl << R2 << endl << P2 << endl;
  
  findRectificationMap(calib_file, out_img_size);

  setCallback(next_video);
  thread th1(imageLoop);
  startGraphics(out_width, out_height);
  th1.join();
  return 0;
}
