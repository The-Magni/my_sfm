#pragma once

#include "opencv2/core/matx.hpp"
#include <array>

class Camera {
    private:
        /* quartenion represent rotation */
        std::array<double, 4> q;
        std::array<double, 3> trans;
        /* intrinsic parameters: f, px, py, k */
        std::array<double, 4> intrinsic;

    public:
        Camera(unsigned int img_width, unsigned int img_height);

        cv::Vec2d project(cv::Vec3d point3D);

        double *getQuartenionParams();

        double *getTranslationParams();

        double *getIntrinsicParams();

        cv::Matx34d getProjectionMat();

        cv::Matx34d getExtrinsicMat();

        cv::Matx33d getIntrinsicMat();

        cv::Vec4d getDistCoeff();

        void updatePose(const cv::Matx33d &R, const cv::Vec3d &t);
};
