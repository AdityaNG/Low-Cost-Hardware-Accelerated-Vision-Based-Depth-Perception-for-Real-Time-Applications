#include "yolo.hpp"

using namespace cv;

cv::Mat frame;

int main(int argc, char **argv) {
    if(argc != 2){
        std::cout << "Usage: ./yolo video_name.mp4\n";
        return 0;
    }


    cv::VideoCapture source(argv[1]);

    initYOLO();

    std::cout << "Starting detection on the video" << argv[1] << "\n";
    while(cv::waitKey(1) < 1){
        source >> frame;
        if (frame.empty()){
            cv::waitKey();
            break;
        }
        
        std::vector<OBJ> temp = processYOLO(frame);
        print(temp);
    }
    return 0;
}