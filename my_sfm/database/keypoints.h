#pragma once

#include "opencv2/core/types.hpp"
#include <sqlite3.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct SerializedKeyPoint {
    float x;
    float y;
    float size;
    float angle;
};

class KeyPointsDB {
    private:
        std::string table_name;
        std::string db_name;
        const int cols = 4;

        void serialize(const std::vector<cv::KeyPoint> &keypoints, std::vector<SerializedKeyPoint> &points_flatten);

        void deserialize(const std::vector<SerializedKeyPoint> &points_flatten, std::vector<cv::KeyPoint> &keypoints);

    public:
        KeyPointsDB(const std::string &db_name);

        bool Insert(unsigned int img_id, const std::vector<cv::KeyPoint> &keypoints);

        bool Retrieve(unsigned int img_id, std::vector<cv::KeyPoint> &keypoints);
};
