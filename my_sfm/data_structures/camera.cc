#include "camera.h"
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include <opencv2/core/quaternion.hpp>
#include <algorithm>

Camera::Camera(unsigned int img_width, unsigned int img_height)
{
    // init intrinsic to arbitrary value
    double f = (double) 1.2 * std::max(img_width, img_height);
    double px = (double) img_width / 2;
    double py = (double) img_height / 2;

    intrinsic[0] = f;
    intrinsic[1] = px;
    intrinsic[2] = py;
    intrinsic[3] = 0;
    // init quartenion to no rotation
    q = {1, 0, 0, 0};
    // init t to all 0
    trans = {0, 0, 0};
}

cv::Vec2d Camera::project(cv::Vec3d point3D)
{
    cv::Vec2d point2D;
    cv::Matx33d R;
    cv::Vec3d t(trans[0], trans[1], trans[2]), point3Dcam;
    cv::Quatd quat(q[0], q[1], q[2], q[3]);
    R = quat.toRotMat3x3();
    double f = intrinsic[0], px = intrinsic[1], py = intrinsic[2], k = intrinsic[3];
    point3Dcam = R * point3D + t;
    point2D[0] = point3Dcam[0] / point3Dcam[2];
    point2D[1] = point3Dcam[1] / point3Dcam[2];
    double r2 = point2D[0] * point2D[0] + point2D[1] * point2D[1];
    double distortion = 1.0 + k * r2;

    point2D[0] = f * distortion * point2D[0] + px;
    point2D[1] = f * distortion * point2D[1] + py;
    return point2D;
}

double *Camera::getQuartenionParams()
{
    return q.data();
}

double *Camera::getIntrinsicParams()
{
    return intrinsic.data();
}

double *Camera::getTranslationParams()
{
    return trans.data();
}

cv::Matx34d Camera::getProjectionMat()
{
    cv::Matx34d P, Rt = getExtrinsicMat();
    cv::Matx33d K = getIntrinsicMat();
    P = K * Rt;
    return P;
}

cv::Matx34d Camera::getExtrinsicMat()
{
    cv::Matx34d Rt;
    cv::Matx33d R;
    cv::Vec3d t(trans[0], trans[1], trans[2]);
    cv::Quatd quat(q[0], q[1], q[2], q[3]);
    R = quat.toRotMat3x3();
    cv::hconcat(R, t, Rt);
    return Rt;
}

cv::Matx33d Camera::getIntrinsicMat()
{
    cv::Matx33d K = cv::Matx33d::eye();
    K(0, 0) = intrinsic[0];
    K(0, 2) = intrinsic[1];
    K(1, 1) = intrinsic[0];
    K(1, 2) = intrinsic[2];
    return K;
}

cv::Vec4d Camera::getDistCoeff()
{
    return cv::Vec4d(intrinsic[3], 0.0, 0.0, 0.0);
}

void Camera::updatePose(const cv::Matx33d &R, const cv::Vec3d &t)
{
    cv::Quatd quat = cv::Quatd::createFromRotMat(R);
    q = {quat.w, quat.x, quat.y, quat.z};

    trans = {t[0], t[1], t[2]};
}
