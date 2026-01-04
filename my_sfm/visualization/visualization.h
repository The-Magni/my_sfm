#pragma once

#include "opencv2/core/mat.hpp"
#include <opencv2/core/affine.hpp>
#include <string>
#include <vector>

class Visualization {
    private:
        std::string points3D_path;
        std::string cameras_path;

        bool readPoints3D(cv::Mat &cloud, cv::Mat &colors);

        bool readCameras(std::vector<cv::Affine3d> &poses, std::vector<cv::Matx33d> &Ks);

    public:
        Visualization(const std::string &dir_path);

        void Process();
};
