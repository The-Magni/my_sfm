#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class Images {
    private:
        std::vector<std::string> img_paths;
    public:
        Images(std::string dir_name);
        cv::Mat loadImages(unsigned int id);
        unsigned int getNumImgs();
};
