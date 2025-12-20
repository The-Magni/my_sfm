#include "optimization.h"
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class ReprojectionError {
    private:
        double observed_x, observed_y;
        cv::Mat K;

    public:
        ReprojectionError(double observed_x, double observed_y, const cv::Mat &K);

        template <typename T>
        bool operator()(const T *const camera, const T *const point, T *residuals) const;
};

ReprojectionError::ReprojectionError(double observed_x, double observed_y, const cv::Mat &K)
{
    this->observed_x = observed_x;
    this->observed_y = observed_y;
    this->K = K;
}

template <typename T>
bool ReprojectionError::operator()(const T *const camera, const T *const point, T *residuals) const {
    Eigen::VectorXd se3(7);
    for (unsigned int i = 0; i < 7; i++) {
        se3(i) = camera[i];
    }
}
