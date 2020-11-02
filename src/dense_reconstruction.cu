//#include <../cfg/cpp/stereo_dense_reconstruction/CamToRobotCalibParamsConfig.h>
//#include <stereo_dense_reconstruction/CamToRobotCalibParamsConfig.h>
//#include "CamToRobotCalibParamsConfig.h"

#include <curl/curl.h>
#include <iostream>
#include <vector>

#include <stdlib.h>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include "elas/elas.h"
#include "nlohmann/json.hpp"
#include <string.h>
#include <math.h>



using json = nlohmann::json;

//#include "popt.h"

//#define CV_CALIB_ZERO_DISPARITY   1024

typedef struct obj {
  std::string name;
  int x;
  int y;
  int h;
  int w;
  int c;
} OBJ;

OBJ obj_list[30];
int obj_list_size = 0;

using namespace cv;
using namespace std;

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
//stereo_dense_reconstruction::CamToRobotCalibParamsConfig config;
FileStorage calib_file;
int debug = 0;
Size out_img_size;
Size calib_img_size;

//image_transport::Publisher dmap_pub;
//ros::Publisher pcl_pub;

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
  for(int j = 0; j<4; j++)
    pos[j] = d_Q[4*j + 0]*x + d_Q[4*j + 1]*y + d_Q[4*j + 2]*d + d_Q[4*j + 3];
    
  double X = pos[0] / pos[3];
  double Y = pos[1] / pos[3];
  double Z = pos[2] / pos[3];

  double point[3];
  for(int j = 0; j<3; j++)
    point[j] = d_XR[3*j + 0]*X + d_XR[3*j + 1]*Y + d_XR[3*j + 2]*Z + d_XT[j];
    
  points[pixelPosition] = make_double3(point[0], point[1], point[2]);
}

void publishPointCloud(Mat& img_left, Mat& dmap) { // CUDAfied
auto start = chrono::high_resolution_clock::now();   
// unsync the I/O of C and C++. 
ios_base::sync_with_stdio(false);

  if (debug == 1) {
    XR = composeRotationCamToRobot(1.3 ,-3.14,1.57);
    XT = composeTranslationCamToRobot(0.0,0.0,0.28);
    cout << "Rotation matrix: " << XR << endl;
    cout << "Translation matrix: " << XT << endl;
  }
    
  int cols = img_left.cols;
  int rows = img_left.rows;
  int totalPixels = cols * rows;

  uchar *d_dmap; // D map needs to be pushed to GPU
  double3 *d_points; // Holds the coordinates of each pixel in 3D space
  double3 *points = (double3*)malloc(sizeof(double3) * totalPixels);
  double *d_XT, *d_XR, *d_Q;

  cudaMalloc(&d_dmap, sizeof(uchar) * totalPixels);
  cudaMalloc(&d_points, sizeof(double3) * totalPixels);
  cudaMalloc(&d_XT, sizeof(double) * 3);
  cudaMalloc(&d_XR, sizeof(double) * 9);
  cudaMalloc(&d_Q, sizeof(double) * 16);

  cudaStream_t s1;
  cudaStreamCreate(&s1);  

  cudaMemcpyAsync(d_dmap, dmap.data, sizeof(uchar) * totalPixels, cudaMemcpyHostToDevice, s1);
  cudaMemcpy(d_XT, XT.data, sizeof(double) * 3, cudaMemcpyHostToDevice);
  cudaMemcpy(d_XR, XR.data, sizeof(double) * 9, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Q, Q.data, sizeof(double) * 16, cudaMemcpyHostToDevice);


  const dim3 blockSize(32, 32, 1);
  const dim3 gridSize((cols / blockSize.x) + 1, (rows / blockSize.y) + 1, 1);

  cudaDeviceSynchronize();

  parallel <<<gridSize, blockSize, 0, s1>>> (d_dmap, d_points, rows, cols, d_XT, d_XR, d_Q);

  cudaDeviceSynchronize();

  cudaStreamDestroy(s1);

  cudaMemcpy(points, d_points, sizeof(double3) * totalPixels, cudaMemcpyDeviceToHost);

  cudaFree(d_points);
  cudaFree(d_dmap);
  cudaFree(d_XR);
  cudaFree(d_XT);
  cudaFree(d_Q);

  auto end = chrono::high_resolution_clock::now();   
  // Calculating total time taken by the program. 
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
  time_taken *= 1e-9;   
  cerr << "Time taken by program is : " << fixed << time_taken << setprecision(9); 
  cerr << " sec\n";

 // cout << "Q matrix: " << Q << "\nimg.size : " << cols <<", " << rows << "\nPOINTS\n";
  /*
  printf("Q matrix: [");
  for(int i = 0; i<4; i++){
    printf("%d, %d, %d, %.17g;\n", Q.data[4*i + 0], Q.data[4*i + 1], Q.data[4*i + 2], Q.data[4*i + 3]);
  }*/

  printf("POINTS\n");
  for (int i = 0; i < img_left.cols; i++){
    for (int j = 0; j < img_left.rows; j++){
      int32_t red, blue, green; 
      red = img_left.at<Vec3b>(j,i)[2];
      green = img_left.at<Vec3b>(j,i)[1];
      blue = img_left.at<Vec3b>(j,i)[0];
      //cout << "[" << points[j*cols + i].x << ";\n " << points[j*cols + i].y << ";\n " << points[j*cols + i].z << "]";      
      printf("[%.17g;\n %.17g;\n %.17g]%u %u %u\n", points[j*cols + i].x, points[j*cols + i].y, points[j*cols + i].z, red, green, blue);   
    }
  }
  printf("POINTS_END\n");
  free(points);
  system ("touch /home/aditya/CODE_Sem3/stereo_dense_reconstruction_no_ros/plotter/3D_maps/reload_check");
  
  if (!dmap.empty()) {
    //sensor_msgs::ImagePtr disp_msg;
    //disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    //dmap_pub.publish(disp_msg);
  }

  //pc.channels.push_back(ch);
  //pcl_pub.publish(pc);
}


void publishPointCloud_OLD(Mat& img_left, Mat& dmap) {
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

  cout << "Q matrix: " << Q << endl;

  cout<<"img.size : "<<img_left.cols<<", "<<img_left.rows<<endl;

  cout << "POINTS"<<endl;
  int BOUNDARY = 50;
   /*
  for (int iObj = 0; iObj < obj_list_size; iObj++)
  {
    int i_lb = constrain(obj_list[iObj].x + obj_list[iObj].w -BOUNDARY, 0, img_left.cols-1), 
    i_ub = constrain(obj_list[iObj].x + obj_list[iObj].w +BOUNDARY, 0, img_left.cols-1), 
    j_lb = constrain(obj_list[iObj].y + obj_list[iObj].h -BOUNDARY, 0, img_left.rows-1), 
    j_ub = constrain(obj_list[iObj].y + obj_list[iObj].h +BOUNDARY, 0, img_left.rows-1);
    
    for (int i = i_lb; i < i_ub; i++) {
      for (int j = j_lb; j < j_ub; j++) {
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

        cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
      }
    } 
  }
  // */

  // /*
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

      cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
    }
  }
  // */
  cout<<"POINTS_END"<<endl;
  
  cout<<"BOXES"<<endl;
  for (int iObj = 0; iObj < obj_list_size; iObj++)
  {
    int i_lb = constrain(obj_list[iObj].x + obj_list[iObj].w/2, 0, img_left.cols-1), 
    i_ub = i_lb + 1, 
    j_lb = constrain(obj_list[iObj].y + obj_list[iObj].h/2, 0, img_left.rows-1), 
    j_ub = j_lb + 1;
    
    for (int i = i_lb; i < i_ub; i++) {
      for (int j = j_lb; j < j_ub; j++) {
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

        cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
      }
    } 
  }
  cout<<"BOXES_END"<<endl;

  system ("touch /home/aditya/CODE_Sem3/stereo_dense_reconstruction_no_ros/plotter/3D_maps/reload_check");
  
  if (!dmap.empty()) {
    //sensor_msgs::ImagePtr disp_msg;
    //disp_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", dmap).toImageMsg();
    //dmap_pub.publish(disp_msg);
  }

  //pc.channels.push_back(ch);
  //pcl_pub.publish(pc);
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
  //rightdpf.convertTo(dmap, CV_8UC1, 4.0);

  imshow("leftdpf", leftdpf);
  imshow("rightdpf", rightdpf);
  
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
void imgCallback(const char* left_img_topic, const char* right_img_topic, const char* left_img_topic_labels) {
  Mat tmpL = imread(left_img_topic, IMREAD_GRAYSCALE);
  Mat tmpR = imread(right_img_topic, IMREAD_GRAYSCALE);
  if (tmpL.empty() || tmpR.empty())
    return;
  
  Mat img_left, img_right, img_left_color, img_left_color_flip;
  img_left_color = imread(left_img_topic_labels);
  flip(img_left_color,img_left_color_flip,1);
  //img_left_color = imread(left_img_topic);

  img_left = tmpL; img_right = tmpR;

  //remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
  //cvtColor(img_left, img_left_color, COLOR_GRAY2BGR);
  
  Mat dmap = generateDisparityMap(img_left, img_right);
  

  //cout<<"D1 : "; int d1 = dmap.at<uchar>(0,0); cout<<d1<<endl;

  publishPointCloud(img_left_color, dmap);
  
  imshow("LEFT_C", img_left_color);
  
  //imshow("LEFT", img_left);
  //imshow("RIGHT", img_right);
  
  imshow("DISP", dmap);
  waitKey(0);
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

void loadYOLO(const char* left_img_labels) {
  std::ifstream input_labels(left_img_labels);
  json json_labels;
  input_labels >> json_labels;

  int iCount=1, jCount=0;
  for (json::iterator it = json_labels.begin(); it != json_labels.end(); ++it) {
    cout << *it << '\n';
    const std::string name = std::string(it.key());
    for (json::iterator jt = (*it).begin(); jt != (*it).end(); ++jt) {
      obj_list[iCount * jCount].name = name;
      obj_list[iCount * jCount].x = (*jt)[0];
      obj_list[iCount * jCount].y = (*jt)[1];
      obj_list[iCount * jCount].h = (*jt)[2];
      obj_list[iCount * jCount].w = (*jt)[3];
      obj_list[iCount * jCount].c = (*jt)[4];

      /*
      cout <<"jt : "<< *jt <<endl;
      cout <<"\t x : "<< (*jt)[0] << '\n';
      cout <<"\t y : "<< (*jt)[1] << '\n';
      cout <<"\t h : "<< (*jt)[2] << '\n';
      cout <<"\t w : "<< (*jt)[3] << '\n';
      cout <<"\t c : "<< (*jt)[4] << '\n';
      cout <<"--------"<<endl;
      */

      jCount++;
      obj_list_size++;
    }
    iCount++;
  }

  cout<<"Got YOLO Objects : " << obj_list_size <<endl;
  for (int i=0; i<obj_list_size; i++) {
    print_OBJ(obj_list[i]);
  }
}





void startVideo() {
  
  //cv::VideoCapture capture("http://192.168.0.109:81/stream", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(320, 240));
  
  //cv::VideoCapture capture("http://192.168.0.109:81/stream", VideoWriter::fourcc('Q', 'V', 'G', 'A'));
  cv::VideoCapture capture("http://192.168.0.109:81/stream", VideoWriter::fourcc('M', 'J', 'P', 'G'));
  //cv::VideoCapture capture("http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4");
  cv::Mat frame;
  while (1) {
      if (!capture.isOpened()) {
          cout<<"Capture failed"<<endl;
          break;
      }

      //Create image frames from capture
      capture >> frame;

      if (!frame.empty()) {
        imshow("stream", frame);
        waitKey(0);
          //do something with your image (e.g. provide it)
          //lastImage = frame.clone();
      }
  }
  
}

/*
R_02: 9.999758e-01 -5.267463e-03 -4.552439e-03 5.251945e-03 9.999804e-01 -3.413835e-03 4.570332e-03 3.389843e-03 9.999838e-01
R_03: 9.995599e-01 1.699522e-02 -2.431313e-02 -1.704422e-02 9.998531e-01 -1.809756e-03 2.427880e-02 2.223358e-03 9.997028e-01

inv(R_02):  18181377685130871000000000/18181818984230806234153769      95489920247678600000000/18181818984230806234153769      83096945849931000000000/18181818984230806234153769 -1053492455566844000000000/200000008826538868575691459  199996081309935949600000000/200000008826538868575691459    677968645189829000000000/200000008826538868575691459 -910487830499633000000000/200000008826538868575691459    -682767004101423200000000/200000008826538868575691459 199996772980057107000000000/200000008826538868575691459

R_23:  0.999557086, 0.022256143, -0.019753071, -0.022226728, 0.99975148, 0.001707184, 0.019786158, -0.001267381, 0.999803488
*/

int main(int argc, char** argv) {

  //startVideo();
  //get_image();

  //return 0;

  //const char* left_img_topic  = "/Users/Shared/KITTI/object/testing/image_0/um_000000.png";
  //const char* right_img_topic = "/Users/Shared/KITTI/object/testing/image_1/um_000000.png";
  const char* calib_file_name = "/home/aditya/CODE_Sem3/stereo_dense_reconstruction_no_ros/calibration/kitti_2011_09_26.yml";
  int calib_width, calib_height, out_width, out_height;
  
  ///*
  calib_width = 1242;
  calib_height = 375;
  out_width = 1242;
  out_height = 375;
  // */

   /*
  calib_width = 1242;
  calib_height = 375;
  out_width = 310;
  out_height = 93;
  // */

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

  //D1 = (Mat_<double>(1,5) << 0, 0, 0, 0, 0);
  //D2 = (Mat_<double>(1,5) << 0, 0, 0, 0, 0);

  cout << " K1 : " << "D1 : " << "R1 : " << "P1 : " << "K2 : " << "D2 : " << "R2 : " << "P2 : " << endl;
  cout <<  K1 << endl << D1 << endl << R1 << endl << P1 << endl << K2 << endl << D2 << endl << R2 << endl << P2 << endl;
  
  findRectificationMap(calib_file, out_img_size);
  char left_img_topic[64];
  char right_img_topic[64];
  char left_img_labels[64];
  char left_img_show_labels[64];

  for (int iImage=0; iImage<5; iImage++) {

    std::strcpy(left_img_topic       , format("/home/aditya/KITTI/object/testing/image_2/%06d.png",  iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
    std::strcpy(right_img_topic      , format("/home/aditya/KITTI/object/testing/image_3/%06d.png",  iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";
    std::strcpy(left_img_labels      , format("/home/aditya/KITTI/output/image_2/%06d.png.json",     iImage).c_str());       //"/Users/Shared/KITTI/output/image_2/000001.png.json";
    std::strcpy(left_img_show_labels , format("/home/aditya/KITTI/output/image_2/%06d.png",          iImage).c_str());            //"/Users/Shared/KITTI/output/image_2/000001.png";
    
    loadYOLO(left_img_labels);
    
    imgCallback(left_img_topic, right_img_topic, left_img_show_labels);
  }
  return 0;
}
