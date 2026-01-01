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

        cv::Vec2d project(cv::Vec3d point3D) const;

        double *getQuartenionParams();

        double *getTranslationParams();

        double *getIntrinsicParams();

        cv::Matx34d getProjectionMat() const;

        cv::Matx34d getExtrinsicMat() const;

        cv::Matx33d getIntrinsicMat() const;

        cv::Vec4d getDistCoeff() const;

        void getPose(cv::Matx33d &R, cv::Vec3d &t) const;

        void updatePose(const cv::Matx33d &R, const cv::Vec3d &t);
};
