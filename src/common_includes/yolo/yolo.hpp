#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "../structs.h"

using namespace cv;


void print(std::vector<OBJ> &objects);

std::vector<OBJ> processYOLO(Mat frame);

void initYOLO(const char *YOLO_CFG, const char* YOLO_WEIGHTS, const char* YOLO_CLASSES);