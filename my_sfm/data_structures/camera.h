#pragma once

#include "opencv2/core/matx.hpp"
#include <array>

class Camera {
    private:
        /* 3 for rotation, 3 for translation */
        std::array<double, 6> extrinsic = {0};
        /* 1 for focal length, 2 for radial distortions */
        std::array<double, 3> intrinsic = {0};
        /* 2 for principal points */
        std::array<double, 2> pp;

    public:
        Camera(unsigned int img_width, unsigned int img_height);

        cv::Vec2d project(cv::Vec3d point3D);

        double *getExtrinsicParams();

        double *getIntrinsicParams();

        double *getPpConst();

        cv::Matx34d getProjectionMat();

        cv::Matx34d getExtrinsicMat();

        cv::Matx33d getIntrinsicMat();

        cv::Vec4d getDistCoeff();

        void updatePose(const cv::Matx33d &R, const cv::Vec3d &t);
};
