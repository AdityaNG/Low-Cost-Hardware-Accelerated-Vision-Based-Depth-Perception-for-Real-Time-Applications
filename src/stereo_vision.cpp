#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <popt.h>
#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <vector>
#include <thread> 
#include <fstream>
#include <future>
#include <filesystem> 

#include "yolo/yolo.hpp"
//#include "elas/elas.h"
#include "graphing/graphing.h"
#include "bayesian/bayesian.h"
#include "common.h"


using namespace cv;
using namespace std;

#define SHOW_VIDEO      // To show the yolo and disparity output as well

#define start_timer(start) auto start = chrono::high_resolution_clock::now();  

#define end_timer(start, var)\
  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(chrono::high_resolution_clock::now() - start).count();\
  time_taken *= 1e-9;\
  var = time_taken;     


void print_OBJ(OBJ o) {
  cout <<"Name : "<< o.name <<endl;
  cout <<"\t x : "<< o.x << '\n';
  cout <<"\t y : "<< o.y << '\n';
  cout <<"\t h : "<< o.h << '\n';
  cout <<"\t w : "<< o.w << '\n';
  cout <<"\t c : "<< o.c << '\n';
  cout <<"--------"<<endl;
}

//////////////////////////////////////// Globals ///////////////////////////////////////////////////////
vector<OBJ> obj_list, pred_list;
Mat XR, XT, Q, P1, P2;
Mat R1, R2, K1, K2, D1, D2, R;
Mat lmapx, lmapy, rmapx, rmapy;
Mat left_img_OLD, right_img_OLD, dmapOLD;
Vec3d T;
FileStorage calib_file;

Size out_img_size;
Size calib_img_size;

int scale_factor = 1; // Modify to change the image resize factor
int point_cloud_extrapolation = 1; // Modify to change the point cloud extrapolation
int input_image_width = 1242, input_image_height = 375; // Default image size in the Kitti dataset
int calib_width, calib_height, out_width, out_height, point_cloud_width, point_cloud_height;

const char* kitti_path;
const char* calib_file_name = "calibration/kitti_2011_09_26.yml";
const char* fsds_calib_file_name = "calibration/fsds.yml";

double pc_t = 0, dmap_t = 0, t_t = 1; // For calculating timings

pthread_t graphicsThread;        // This is the openGL thread that plots the points in 3D)
bool graphicsBeingUsed = true;   // To know if graphics is used or not
bool objectTracking = true;      // To know if object tracking is being done
      
int debug = 0;                   // Applies different roation and translation to the points
int frame_skip = 1;              // Skip by frame_skip frames
int video_mode = 0;              // Loop among all the images in the given directory
bool draw_points = true;         // Render the points in 3D
bool graphicsThreadExit = false; // The graphics thread toggles this when it exits

// Graphics
extern int Oindex;

double *points; // Holds the coordinates of each pixel in 3D space
uchar *color = NULL;
////////////////////////////////////////////////////////////////////////////////////////////////////////

void Init(){		
	if (debug == 1) {
		//XR = Mat_<double>(3,1) << 1.3 , -3.14, 1.57;
		XR = Mat_<double>(3,1) << M_PI/3, 0, 0; //M_PI
		XT = Mat_<double>(3,1) << -4, -1.0, 1.7;
		cout << "Rotation matrix: " << XR << endl;
		cout << "Translation matrix: " << XT << endl;
	}
	points = (double*)malloc(sizeof(double) * 3 * point_cloud_width * point_cloud_height);
}

void clean(){
  free(points);
}

int constrain(int a, int lb, int ub) {
       if(a < lb) return lb;
  else if(a > ub) return ub;
           else return a;
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

string YOLO_to_KITTI_labels(string y) {
  // '', 'Van', '', '', 'Person_sitting', '', '', '' or 'DontCare'
  if (y == "car") {
    return "Car";
  } else if (y == "truck") {
    return "Truck";
  } else if (y == "person") {
    return "Pedestrian";
  } else if (y == "bicycle") {
    return "Cyclist";
  } else if (y == "train") {
    return "Tram";
  }

  return "Misc";
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
void publishPointCloud(Mat& img_left, Mat& dmap, char* OUT_img_topic=NULL) {    
	start_timer(pc_start); 
  int pointCloudCol;
	Mat V = Mat(4, 1, CV_64FC1);
	Mat pos = Mat(4, 1, CV_64FC1);


	if (draw_points) {
		for (int j = 0; j < img_left.rows; j++) {
      pointCloudCol = 0;
			for (int i = 0; i < img_left.cols; ++i) {
				int d = dmap.at<uchar>(j,i);
				//cout<<d<<endl;
				// if low disparity, then ignore
				/*
				if (d < 2) {
				  continue;
				}*/
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
				//Mat point3d_robot = XR * point3d_cam + XT;
				//points.push_back(Point3d(point3d_robot));
			  	//ch.values.push_back(*reinterpret_cast<float*>(&rgb));
				points[j*point_cloud_width + pointCloudCol + 0] = X;
				points[j*point_cloud_width + pointCloudCol + 1] = Y;
				points[j*point_cloud_width + pointCloudCol + 2] = Z;
        pointCloudCol += 3;
    			//cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
    		}
  		}
  	}

  if(objectTracking){
		for(auto& object : obj_list) {
			int i_lb = constrain(object.x, 0, img_left.cols-1), 
			    i_ub = constrain(object.x + object.w, 0, img_left.cols-1), 
			    j_lb = constrain(object.y, 0, img_left.rows-1), 
			    j_ub = constrain(object.y + object.h, 0, img_left.rows-1);
			double X=0, Y=0, Z=0;
			
			for(int i = i_lb; i < i_ub; i++) {
        pointCloudCol = 0;
				for(int j = j_lb; j < j_ub; j++) {   
					X += points[j*out_width + pointCloudCol + 0];  
					Y += points[j*out_width + pointCloudCol + 1];     
					Z += points[j*out_width + pointCloudCol + 2];  
          pointCloudCol += 3;
				}
			} 
			if(graphicsBeingUsed) appendOBJECTS(X/((i_ub-i_lb)*(j_ub-j_lb)), Y/((i_ub-i_lb)*(j_ub-j_lb)), Z/((i_ub-i_lb)*(j_ub-j_lb)), object.r, object.g, object.b); 
		}
	}
  end_timer(pc_start, pc_t);
  //cout << "pc : " << fixed << time_taken << setprecision(9); 
  //cout << " \t";
  //cout<<"POINTS_END\nBOXES\n";
//  FILE * pFile;
//
//  if (video_mode==3) {
//    // TODO : Open file
//    pFile = fopen(OUT_img_topic,"w");
//  }
//
//  for (auto& object : obj_list) {
//    //print_OBJ(object);
//    /*
//    int i_lb = constrain(object.x + object.w/2, 0, img_left.cols-1), 
//    i_ub = i_lb + 1, 
//    j_lb = constrain(object.y + object.h/2, 0, img_left.rows-1), 
//    j_ub = j_lb + 1;
//    */
//
//    int i_lb = constrain(object.x, 0, img_left.cols-1), 
//    i_ub = constrain(object.x + object.w, 0, img_left.cols-1), 
//    j_lb = constrain(object.y, 0, img_left.rows-1), 
//    j_ub = constrain(object.y + object.h, 0, img_left.rows-1);
//    
//    double X=0, Y=0, Z=0;
//
//    for (int i = i_lb; i < i_ub; i++) {
//      for (int j = j_lb; j < j_ub; j++) {
//        int d = dmap.at<uchar>(j,i);
//        /*
//        if (d < 2) {
//          continue;
//        }
//        */
//        V.at<double>(0,0) = (double)(i);
//        V.at<double>(1,0) = (double)(j);
//        V.at<double>(2,0) = (double)d;
//        V.at<double>(3,0) = 1.;
//        
//        pos = Q * V; // 3D homogeneous coordinate
//        X += pos.at<double>(0,0) / pos.at<double>(3,0);
//        Y += pos.at<double>(1,0) / pos.at<double>(3,0);
//        Z += pos.at<double>(2,0) / pos.at<double>(3,0);
//        Mat point3d_cam = Mat(3, 1, CV_64FC1);
//        point3d_cam.at<double>(0,0) = X;
//        point3d_cam.at<double>(1,0) = Y;
//        point3d_cam.at<double>(2,0) = Z;
//        // transform 3D point from camera frame to robot frame
//        Mat point3d_robot = XR * point3d_cam + XT;
//        points.push_back(Point3d(point3d_robot));        
//      }
//    }
//    appendOBJECTS(X/((i_ub-i_lb)*(j_ub-j_lb)), Y/((i_ub-i_lb)*(j_ub-j_lb)), Z/((i_ub-i_lb)*(j_ub-j_lb)), object.r, object.g, object.b); 
//
//    if (video_mode==3) {
//      int w_3d=0, h_3d=0, l_3d=0;
//      double rot_y_3d=0;
//      // 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
//      printf("%s 0.00 0 0.0 %d %d %d %d %d %d %d %f %f %f %f\n", YOLO_to_KITTI_labels(object.name).c_str(), object.x, object.y, object.x+object.w, object.y+object.h, w_3d, h_3d, l_3d, X, Y, Z, rot_y_3d);
//      fprintf(pFile, "%s 0.00 0 0.0 %d %d %d %d %d %d %d %f %f %f %f\n", YOLO_to_KITTI_labels(object.name).c_str(), object.x, object.y, object.x+object.w, object.y+object.h, w_3d, h_3d, l_3d, X, Y, Z, rot_y_3d);
//    }
//  }
//
//  if (video_mode==3) {
//    // TODO : Close file
//    fclose (pFile);
//  }
//  //cout<<"BOXES_END\n";
//
//  //system ("touch ../plotter/3D_maps/reload_check");
//  
//  if (!dmap.empty()) {
//    // TODO : Do something
//  }
//  //updateGraph();
//  auto end = chrono::high_resolution_clock::now();   
//  // Calculating total time taken by the program. 
//  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
//  time_taken *= 1e-9;   
//  pc_t = time_taken;
}

//Mat depth_map;
//void publishPointCloud_KITTI(Mat& img_left, Mat& dmap) {
//
//  if (img_left.empty() || dmap.empty()) {
//    printf("(empty)\t");
//    return;
//  }
//      
//  
//  Mat V = Mat(4, 1, CV_64FC1);
//  Mat pos = Mat(4, 1, CV_64FC1);
//  vector< Point3d > points;
//
//  //if (draw_points) {
//  for (int i = 0; i < img_left.cols; i++) {
//    for (int j = 0; j < img_left.rows; j++) {
//      int d = dmap.at<uchar>(j,i);
//      //cout<<d<<endl;
//      // if low disparity, then ignore
//      /*
//      if (d < 2) {
//        continue;
//      }*/
//      // V is the vector to be multiplied to Q to get
//      // the 3D homogenous coordinates of the image point
//      V.at<double>(0,0) = (double)(i);
//      V.at<double>(1,0) = (double)(j);
//      V.at<double>(2,0) = (double)d;
//      V.at<double>(3,0) = 1.;
//      
//      pos = Q * V; // 3D homogeneous coordinate
//      double X = pos.at<double>(0,0) / pos.at<double>(3,0);
//      double Y = pos.at<double>(1,0) / pos.at<double>(3,0);
//      double Z = pos.at<double>(2,0) / pos.at<double>(3,0);
//      Mat point3d_cam = Mat(3, 1, CV_64FC1);
//      point3d_cam.at<double>(0,0) = X;
//      point3d_cam.at<double>(1,0) = Y;
//      point3d_cam.at<double>(2,0) = Z;
//      // transform 3D point from camera frame to robot frame
//      Mat point3d_robot = XR * point3d_cam + XT;
//      points.push_back(Point3d(point3d_robot));
//      
//      int32_t red, blue, green;
//      red = img_left.at<Vec3b>(j,i)[2];
//      green = img_left.at<Vec3b>(j,i)[1];
//      blue = img_left.at<Vec3b>(j,i)[0];
//      //ch.values.push_back(*reinterpret_cast<float*>(&rgb));
//
//      //appendPOINT(X, Y, Z, red/255.0, green/255.0, blue/255.0);
//
//      //if (video_mode == 2) {
//        //depth_map.data[j*img_left.cols + i] = (uint16_t) Z * 256;
//        //depth_map.at<uint16_t>(Point(j,i)) = 255;//(uint16_t) Z * 256;
//        //depth_map.at<uint16_t>(Point(j,i)) = 255;//(uint16_t) Z * 256;
//        //depth_map.at<uchar>(j, i) = 250;
//        //depth_map.at<uint16_t>(j, i) *= 250;
//        
//        //Mat_<uchar>& M1 = (Mat_<uchar>&)depth_map;
//        //M1.at<uchar>(Point(j,i)) = 255;
//
//        //printf(" (%d)", depth_map.at<uchar>(j, i));
//
//        //depth_map.at<unsigned short>(j, i) = (unsigned short) Z * 255;
//        depth_map.at<unsigned short>(j, i) = (unsigned short) Z * 255;
//
//        //depth_map.at<uchar>(j, i) = (unsigned short) 250*250;//Z * 256;
//
//        //printf(" -> (%d)\n----", depth_map.at<uchar>(j, i));
//
//      //}
//
//      //cout<<point3d_robot<< red << " " << green << " " << blue <<endl;
//    }
//  }
//  //}
//  
//  
//  if (!dmap.empty()) {
//    // TODO : Do something
//  }
//  //updateGraph();
//  auto end = chrono::high_resolution_clock::now();   
//  // Calculating total time taken by the program. 
//  double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
//  time_taken *= 1e-9;   
//  pc_t = time_taken;
//}
//
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
	Oindex = 0;
  	if (left.empty() || right.empty()) {
		  printf("Image empty\n");
		  return left;
	  } 
  	const Size imsize = left.size();
  	const int32_t dims[3] = {imsize.width, imsize.height, imsize.width};
  	Mat leftdpf = Mat::zeros(imsize, CV_32F);
  	Mat rightdpf = Mat::zeros(imsize, CV_32F);

	//static Elas::parameters param(Elas::MIDDLEBURY);//param(Elas::ROBOTICS);
	//static int res = printf("Post Process only left = %d\n", param.postprocess_only_left = true);//false;
	//static Elas elas(param);
  	
	//elas.process(left.data, right.data, leftdpf.ptr<float>(0), rightdpf.ptr<float>(0), dims);
	Mat dmap = Mat(out_img_size, CV_8UC1, Scalar(0));
	
	leftdpf.convertTo(dmap, CV_8UC1, 4.0);	
	return dmap;
}

/*
 * Function:  imgCallback_video 
 * --------------------
 * Loads the input images into Mats
 * Undistorts and Rectifies the images with remap()
 * Generates disparity map with generateDisparityMap(img_left, img_right)
 *
 *  const char* left_img_topic: path to left image
 *  const char* right_img_topic: path to right image
 *  returns: void
 *
 */
void imgCallback_video() {
	Mat left_img = left_img_OLD; Mat right_img = right_img_OLD, img_left, img_right;
	if (left_img.empty() || right_img.empty()) return;

	cvtColor(left_img, img_left, COLOR_BGRA2GRAY);
	cvtColor(right_img, img_right, COLOR_BGRA2GRAY);

	//remap(tmpL, img_left, lmapx, lmapy, cv::INTER_LINEAR); remap(tmpR, img_right, rmapx, rmapy, cv::INTER_LINEAR);
  
	start_timer(dmap_start);   
	dmapOLD = generateDisparityMap(img_left, img_right);  
	end_timer(dmap_start, dmap_t);
}

/*
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
*/

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

	K1.at<double>(0, 0) /= scale_factor; 
	K1.at<double>(0, 1) /= scale_factor;
	K1.at<double>(0, 2) /= scale_factor; 
	K1.at<double>(1, 0) /= scale_factor;
	K1.at<double>(1, 1) /= scale_factor;
	K1.at<double>(1, 2) /= scale_factor; 

	K2.at<double>(0, 0) /= scale_factor; 
	K2.at<double>(0, 1) /= scale_factor;
	K2.at<double>(0, 2) /= scale_factor; 
	K2.at<double>(1, 0) /= scale_factor;
	K2.at<double>(1, 1) /= scale_factor;
	K2.at<double>(1, 2) /= scale_factor; 

	cout << "Scaled K1: " << K1 << '\n';
	cout << "Scaled K2: " << K2 << '\n';

	// Divide K1 and K2's first two rows with scale factor

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

	stereoRectify(K1, D1, K2, D2, calib_img_size, R, Mat(T), R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 0, finalSize, &validRoi[0], &validRoi[1]);

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

/*
std::size_t number_of_files_in_directory(fs::path path)
{
    using fs::directory_iterator;
    using fp = bool (*)( const fs::path&);
    return std::count_if(directory_iterator(path), directory_iterator{}, (fp)fs::is_regular_file);
}


void next() {
  printf("NEXT Called\n");
  static int iImage=0;
  if (video_mode == 1) {
    char left_img_topic[128];
    char right_img_topic[128];
    char right_img_dir[128];
    std::strcpy(right_img_dir , format("%svideo/testing/image_03/%04d/", kitti_path, iImage).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";
    fs::path path_to_folder(right_img_dir);
    size_t max_files = number_of_files_in_directory(path_to_folder);

    Mat left_img, right_img, dmap, YOLOL_Color, img_left_color_flip;
    //thread th1(imgCallback_video);
    thread th1;

    play_video = 0;
    //while (play_video)
    while (1){      
      for (int iFrame = 0; iFrame < max_files; iFrame++)
      {
        while (!play_video) {}

        if (t_t!=0)
          printf("(FPS=%f) ", 1/t_t);
           
        
        
        std::strcpy(left_img_topic  , format("%s/video/testing/image_02/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
        std::strcpy(right_img_topic , format("%s/video/testing/image_03/%04d/%06d.png", kitti_path, iImage, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";

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

        if ( iFrame%frame_skip == 0) {
          printf("(DISP) \t ");
          //imgCallback_video(left_img, right_img, dmap);
          left_img_OLD = left_img.clone();
          right_img_OLD = right_img.clone();

          //disp_parallel = std::async(imgCallback_video);
          th1 = thread(imgCallback_video);
        }
          
        if (iFrame%frame_skip == frame_skip-1) {
          printf("(JOIN) \t");
          th1.join();
          dmap = dmapOLD.clone();
        }

        printf("(%d, %d) ", dmap.rows, dmap.cols);
        
        //if (iFrame%frame_skip == frame_skip-1) {
        // th1.join();
        //  dmap = dmapOLD.clone();
        //}
    

        //obj_list = f.get(); // Getting obj_list from the future object which the async call return to f
        publishPointCloud(left_img, dmap);
        printf("(PC Done) ");
        updateGraph();

        if (1) {
          flip(YOLOL_Color, img_left_color_flip,1);
          //flip(left_img, img_left_color_flip,1);
          
          imshow("LEFT_C", img_left_color_flip);
          //imshow("DISP", dmap);
          
          //waitKey(1);
        }

        auto end = chrono::high_resolution_clock::now();   
        // Calculating total time taken by the program. 
        double time_taken =  chrono::duration_cast<chrono::nanoseconds>(end - start).count(); 
        time_taken *= 1e-9;   
        t_t = time_taken;
        
        printf("(t_t=%f, \t yd_t=%f, \t pc_t=%f)\n",t_t, yd_t, pc_t);
      }
    }
  } else if (video_mode == 3) {
    char OUT_img_topic[128];
    
    char left_img_topic[128];
    char right_img_topic[128];
    char right_img_dir[128];
    std::strcpy(right_img_dir , format("%svideo/testing/image_03/%04d/", kitti_path, iImage).c_str());    // "/Users/Shared/KITTI/object/testing/image_3/000001.png";
    fs::path path_to_folder(right_img_dir);
    size_t max_files = number_of_files_in_directory(path_to_folder);

    Mat left_img, right_img, dmap, YOLOL_Color, img_left_color_flip;
    //thread th1(imgCallback_video);
    thread th1;

    play_video = 0;
    //while (play_video)
    while (1)
    {

      
      for (int iFrame = 0; iFrame < max_files; iFrame++)
      {
        while (!play_video) {}

        if (t_t!=0)
        printf("(FPS=%f) ", 1/t_t);
           
        
        
        std::strcpy(OUT_img_topic  , format("%sOUTPUT_3D/%06d.txt", kitti_path, iFrame).c_str());  
        std::strcpy(left_img_topic  , format("%s/threeD/training/image_2/%06d.png", kitti_path, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
        std::strcpy(right_img_topic , format("%s/threeD/training/image_3/%06d.png", kitti_path, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";

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

        if ( iFrame%frame_skip == 0) {
          printf("(DISP) \t ");
          //imgCallback_video(left_img, right_img, dmap);
          left_img_OLD = left_img.clone();
          right_img_OLD = right_img.clone();

          //disp_parallel = std::async(imgCallback_video);
          th1 = thread(imgCallback_video);
        }
          
        if (iFrame%frame_skip == frame_skip-1) {
          printf("(JOIN) \t");
          th1.join();
          dmap = dmapOLD.clone();
        }

        printf("(%d, %d) ", dmap.rows, dmap.cols);
        
        //if (iFrame%frame_skip == frame_skip-1) {
        // th1.join();
        //  dmap = dmapOLD.clone();
        //}
    

        //obj_list = f.get(); // Getting obj_list from the future object which the async call return to f
        publishPointCloud(left_img, dmap, OUT_img_topic);
        printf("(PC Done) ");
        updateGraph();

        if (0) {
          flip(YOLOL_Color, img_left_color_flip,1);
          //flip(left_img, img_left_color_flip,1);
          
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
  } 
  else if (video_mode == 2) { 
    // KITTI Eval Code

    char OUT_img_topic[128];
    char left_img_topic[128];
    char right_img_topic[128];
    char right_img_dir[128];
    std::strcpy(right_img_dir , format("%s2011_09_26/2011_09_26_drive_0001_sync/image_03/data", kitti_path).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";
    printf("DIR=%s\n", right_img_dir);
    fs::path path_to_folder(right_img_dir);
    size_t max_files = 99;//number_of_files_in_directory(path_to_folder);

    Mat left_img, right_img, dmap, YOLOL_Color, img_left_color_flip;
    //thread th1(imgCallback_video);
    thread th1;

    play_video = 0;
    depth_map = Mat::zeros(out_img_size, CV_16UC1);

    //line(depth_map,Point(0,0), Point(out_width, out_width),Scalar(255,255,255),1,8,0);


    //while (play_video)
    while (1)
    {
      for (int iFrame = 5; iFrame < max_files; iFrame++)
      {
        // while (!play_video) {}

        if (t_t!=0)
          printf("(FPS=%f) ", 1/t_t);
           
        
        
        //000 000 000 0
        std::strcpy(OUT_img_topic  , format("%sOUTPUT/%010d.png", kitti_path, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
        std::strcpy(left_img_topic  , format("%s2011_09_26/2011_09_26_drive_0001_sync/image_02/data/%010d.png", kitti_path, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_2/000001.png";
        std::strcpy(right_img_topic , format("%s2011_09_26/2011_09_26_drive_0001_sync/image_03/data/%010d.png", kitti_path, iFrame).c_str());    //"/Users/Shared/KITTI/object/testing/image_3/000001.png";

        printf("(file=%s) \n", left_img_topic);

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

        //if ( iFrame%frame_skip == 0) {
          printf("(DISP) \t ");
          //imgCallback_video(left_img, right_img, dmap);
          left_img_OLD = left_img.clone();
          right_img_OLD = right_img.clone();

          //disp_parallel = std::async(imgCallback_video);
          th1 = thread(imgCallback_video);
        //}
          
        //if (iFrame%frame_skip == frame_skip-1) {
          printf("(JOIN) \t");
          th1.join();
          dmap = dmapOLD.clone();
          //depth_map = dmapOLD.clone();
          
        //}

        printf("(%d, %d) ", dmap.rows, dmap.cols);
        
        //if (iFrame%frame_skip == frame_skip-1) {
        // th1.join();
        //  dmap = dmapOLD.clone();
        //}
    

        //obj_list = f.get(); // Getting obj_list from the future object which the async call return to f
        publishPointCloud_KITTI(left_img, dmap);
        printf("(PC Done) ");
        updateGraph();

        //depth_map.convertTo(depth_map, CV_8U, 255); // 8bit to 16
        //imshow("depth", depth_map); waitKey(0);
        imwrite(OUT_img_topic, depth_map);

        if (0) {
          flip(YOLOL_Color, img_left_color_flip,1);
          //flip(left_img, img_left_color_flip,1);
          
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
  play_video = !play_video;
}

void imageLoop() {
  while (1) next();
}
*/
unsigned fileCounter(string path){
	auto dirIter = std::filesystem::directory_iterator(path);	
	unsigned fileCount = std::count_if(begin(dirIter), end(dirIter), [](auto& entry) { return entry.is_regular_file();});
	return fileCount;
}

void imageLoop(){
	unsigned iImage = 0;
	char left_img_topic[128], right_img_topic[128];    
	size_t max_files = fileCounter(format("%s/video/testing/image_02/%04u/", kitti_path, iImage)); 
	printf("Max files = %lu\n", max_files);
	Mat left_img, right_img, YOLOL_Color, img_left_color_flip, rgba;
	//thread skipThread;

	for(unsigned iFrame = 0; (iFrame < max_files) && !graphicsThreadExit; iFrame++){            
		start_timer(t_start);        
		strcpy(left_img_topic , format("%s/video/testing/image_02/%04u/%06u.png", kitti_path, iImage, iFrame).c_str());    
		strcpy(right_img_topic, format("%s/video/testing/image_03/%04u/%06u.png", kitti_path, iImage, iFrame).c_str());    

		left_img = imread(left_img_topic, IMREAD_UNCHANGED);
		right_img = imread(right_img_topic, IMREAD_UNCHANGED);

		resize(left_img, left_img_OLD, out_img_size);

		YOLOL_Color = left_img_OLD.clone();	
		if(objectTracking){
			//auto f = std::async(std::launch::async, processYOLO, YOLOL_Color); // Asynchronous call to YOLO 	
			resize(right_img, right_img_OLD, out_img_size);   
			imgCallback_video();	
			cvtColor(left_img, rgba, cv::COLOR_BGR2BGRA);
			color = (uchar*)rgba.ptr<unsigned char>(0);
			//obj_list = f.get(); // Getting obj_list from the future object which the async call returned to f
			//pred_list = get_predicted_boxes(); // Bayesian
			//append_old_objs(obj_list);
			//obj_list.insert( obj_list.end(), pred_list.begin(), pred_list.end() );
		}	
		else{
			resize(right_img, right_img_OLD, out_img_size);   
			imgCallback_video();	
			cvtColor(left_img, rgba, cv::COLOR_BGR2BGRA);
			color = (uchar*)rgba.ptr<unsigned char>(0);
		}	
		publishPointCloud(left_img, dmapOLD);    
		end_timer(t_start, t_t);	
		#ifdef SHOW_VIDEO
			//flip(left_img, img_left_color_flip,1);
			imshow("Detections", YOLOL_Color);
			imshow("Disparity", dmapOLD);
			waitKey(video_mode);
		#endif        
		printf("(FPS=%f) (%d, %d) (t_t=%f, dmap_t=%f, pc_t=%f)\n", 1/t_t, dmapOLD.rows, dmapOLD.cols, t_t, dmap_t, pc_t);
	}
}

int main(int argc, const char** argv) {
	ios_base::sync_with_stdio(false);
	static struct poptOption options[] = { 
	  { "kitti_path", 'k', POPT_ARG_STRING, &kitti_path, 0, "Path to KITTI Dataset", "STR" },
	  { "video_mode", 'v', POPT_ARG_INT, &video_mode, 0, "Set v=1 Kitti video mode", "NUM" },
	  { "draw_points", 'p', POPT_ARG_SHORT, &draw_points, 0, "Set p=1 to plot out points", "NUM" },
	  { "frame_skip", 'f', POPT_ARG_INT, &frame_skip, 0, "Set frame_skip to skip disparity generation for f frames (Not yet implemented)", "NUM" },
	  { "debug", 'd', POPT_ARG_INT, &debug, 0, "Set d=1 for cam to robot frame calibration", "NUM" },
	  { "object_tracking", 't', POPT_ARG_SHORT, &objectTracking, 0, "Set t=1 for enabling object tracking", "NUM" },
	  { "input_image_width", 'w', POPT_ARG_INT, &frame_skip, 0, "Set the input image width (default value is 1242, i.e Kitti image width)", "NUM" },
	  { "input_image_height", 'h', POPT_ARG_INT, &frame_skip, 0, "Set the input image height (default value is 375, i.e Kitti image height)", "NUM" },		{ "scale_factor", 's', POPT_ARG_INT, &scale_factor, 0, "All operations will be applied after shrinking the image by this factor", "NUM" },
	  { "extrapolate_point_cloud", 'e', POPT_ARG_INT, &point_cloud_extrapolation, 0, "Extrapolate the point cloud by this factor", "NUM" },
	  POPT_AUTOHELP
	  { NULL, 0, 0, NULL, 0, NULL, NULL }
	};	
	poptContext poptCONT = poptGetContext("main", argc, argv, options, POPT_CONTEXT_KEEP_FIRST);	
	if (argc < 2) {
		poptPrintUsage(poptCONT, stderr, 0);
		exit(1);
	}
	int c; 
	while((c = poptGetNextOpt(poptCONT)) >= 0);
	if (c < -1) { // An error occurred during option processing 
	    fprintf(stderr, "stereo_vision: %s -- \'%s\'\n",
	            poptStrerror(c), poptBadOption(poptCONT, POPT_BADOPTION_NOALIAS));
	    poptPrintUsage(poptCONT, stderr, 0);
	    return 1;
	}	
	if(objectTracking){
		printf("** Object Tracking enabled\n");
		//initYOLO("src/yolo/yolov4-tiny.cfg", "src/yolo/yolov4-tiny.weights", "src/yolo/classes.txt");
	} 
	else printf("** Object tracking disabled\n"); 	
	printf("KITTI Path: %s \n", kitti_path);

	calib_width = input_image_width;
	calib_height = input_image_height;
	out_width = input_image_width/scale_factor;
	out_height = input_image_height/scale_factor;
	point_cloud_width = out_width * point_cloud_extrapolation;
	point_cloud_height = out_height * point_cloud_extrapolation;	
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
	Init();
	int ret = pthread_create(&graphicsThread, NULL, startGraphics, NULL);
	if(ret){
		fprintf(stderr, "Graphics thread could not be launched.\npthread_create : %s\n", strerror(ret));
		exit(-1);
	}

	#ifdef SHOW_VIDEO
		namedWindow("Detections", cv::WINDOW_NORMAL); // Needed to allow resizing of the image shown
		namedWindow("Disparity", cv::WINDOW_NORMAL); // Needed to allow resizing of the image shown
	#endif

	imageLoop();
	printf("imageLoop exitted...\n");
	clean();
	return 0;
}
  /*
  for (int i=0; i<10; i++) {
    std::vector<OBJ> tmp_list;
    
    OBJ temp;
    temp.name = "tmp";
    temp.x = 3*i;
    temp.y = 4*i+1;
    temp.w = 1;
    temp.h = 1;
    temp.c = 0.1;
    temp.g = 255.0;
    temp.b = 255.0;
    temp.r = 255.0;
    tmp_list.push_back(temp);

    temp.x = -3*i - 80;
    temp.y = 5*i + 70;
    tmp_list.push_back(temp);

    append_old_objs(tmp_list);
  }
  
  display_history();

  int x=0;
  int y=0;
  predict(0, &x, &y);
  printf("Prediction 0 : (%d, %d)\n", x, y);
  predict(1, &x, &y);
  printf("Prediction 1 : (%d, %d)\n", x, y);

  return 0;

  // 
  
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
  out_width = 1242 /4 ;
  out_height = 375 /4;
  
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
*/