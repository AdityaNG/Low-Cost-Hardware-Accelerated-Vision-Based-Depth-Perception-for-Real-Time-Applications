#include "yolo.hpp"

constexpr float CONFIDENCE_THRESHOLD = 0.5;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

static cv::dnn::Net net;
static std::vector<String> output_names;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors)/sizeof(colors[0]);
std::vector<std::string> class_names;


void print(std::vector<OBJ> &objects){
    std::cout << "\n{\n";
    for (auto& object : objects){
        std::cout << '[' << object.name << '(' << object.x << ',' << object.y << ',' << object.w << ',' << object.h << ',' << object.c << ")]\n";
    }
    std::cout << "\n}\n";
}

std::vector<OBJ> processYOLO(Mat frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
    std::vector<cv::Mat> detections;
    std::vector<OBJ> objects; // Detected objects
    auto total_start = std::chrono::steady_clock::now();

    net.setInput(blob);   

    auto dnn_start = std::chrono::steady_clock::now();
    net.forward(detections, output_names);
    auto dnn_end = std::chrono::steady_clock::now();

    std::vector<int> indices[NUM_CLASSES];
    std::vector<cv::Rect> boxes[NUM_CLASSES];
    std::vector<float> scores[NUM_CLASSES];

    for (auto& output : detections){
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++){
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width/2, y - height/2, width, height);

            for (int c = 0; c < NUM_CLASSES; c++){
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD){
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++) cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        
    for (int c= 0; c < NUM_CLASSES; c++){
        for (size_t i = 0; i < indices[c].size(); ++i){
            const auto color = colors[c % NUM_COLORS];

            auto idx = indices[c][i];
            const auto& rect = boxes[c][idx];
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 1);

            std::ostringstream label_ss;
            label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
            auto label = label_ss.str();
                
            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));

            OBJ temp;
            temp.name = class_names[c];
            //strcpy(temp.name, class_names[c]);
            temp.x = rect.x;
            temp.y = rect.y;
            temp.w = rect.width;
            temp.h = rect.height;
            temp.c = scores[c][idx];
            temp.g = color[0] / 255.0;
            temp.b = color[1] / 255.0;
            temp.r = color[2] / 255.0;
            objects.push_back(temp);
        }
    }
    
    auto total_end = std::chrono::steady_clock::now();

    float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
    float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    std::ostringstream stats_ss;
    stats_ss << std::fixed << std::setprecision(2);
    stats_ss << "Total FPS: " << total_fps << ", Inference FPS: " << inference_fps;
    auto stats = stats_ss.str();
         
    int baseline;
    auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
    cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));
    return objects;
}

void initYOLO(const char *YOLO_CFG, const char* YOLO_WEIGHTS, const char* YOLO_CLASSES) {
    std::ifstream class_file(YOLO_CLASSES);
    if (!class_file){
            std::cerr << "failed to open classes.txt\n";
            exit(-1);
    }

    net = cv::dnn::readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS);
    output_names = net.getUnconnectedOutLayersNames();
    
    std::string line;
    while (std::getline(class_file, line)) class_names.push_back(line);

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    std::cout<<"YOLO Init done\n";
}
