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

using namespace cv;

typedef struct object {
  std::string name; // Name of the detection
  int x, y; // Coordinates
  int w, h; // Width and height
  float c; // Confidence
} OBJ;

void print(std::vector<OBJ> &objects);

std::vector<OBJ> processYOLO(Mat frame);

void initYOLO();