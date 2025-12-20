#pragma once

#include <sqlite3.h>
#include <opencv2/opencv.hpp>
#include <string>

class DescriptorsDB {
    private:
        std::string table_name;
        std::string db_name;

    public:
        DescriptorsDB(std::string db_name);

        bool Insert(unsigned int img_id, const cv::Mat &descriptors);

        bool Retrieve(unsigned int img_id, cv::Mat &descriptors);
};
