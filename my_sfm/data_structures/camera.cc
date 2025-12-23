#include "camera.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include <algorithm>

Camera::Camera(unsigned int img_width, unsigned int img_height)
{
    double f = (double) 1.2 * std::max(img_width, img_height);
    double px = (double) img_width / 2;
    double py = (double) img_height / 2;

    intrinsic[0] = f;
    pp[0] = px;
    pp[1] = py;
}

cv::Vec2d Camera::project(cv::Vec3d point3D)
{
    cv::Vec2d point2D;
    cv::Matx33d R;
    cv::Vec3d rvec(extrinsic[0], extrinsic[1], extrinsic[2]), t(extrinsic[3], extrinsic[4], extrinsic[5]), point3Dcam;
    cv::Rodrigues(rvec, R);
    double f = intrinsic[0], px = pp[0], py = pp[1], l1 = intrinsic[1], l2 = intrinsic[2];
    point3Dcam = R * point3D + t;
    point2D[0] = point3Dcam[0] / point3Dcam[2];
    point2D[1] = point3Dcam[1] / point3Dcam[2];
    double r2 = point2D[0] * point2D[0] + point2D[1] * point2D[1];
    double distortion = 1.0 + l1 * r2 + l2 * r2 * r2;

    point2D[0] = f * distortion * point2D[0] + px;
    point2D[1] = f * distortion * point2D[1] + py;
    return point2D;
}

double *Camera::getExtrinsicParams()
{
    return extrinsic.data();
}

double *Camera::getIntrinsicParams()
{
    return intrinsic.data();
}

double *Camera::getPpConst()
{
    return pp.data();
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
    cv::Vec3d rvec(extrinsic[0], extrinsic[1], extrinsic[2]), t(extrinsic[3], extrinsic[4], extrinsic[5]);
    cv::Rodrigues(rvec, R);
    cv::hconcat(R, t, Rt);
    return Rt;
}

cv::Matx33d Camera::getIntrinsicMat()
{
    cv::Matx33d K = cv::Matx33d::eye();
    K(0, 0) = intrinsic[0];
    K(0, 2) = pp[0];
    K(1, 1) = intrinsic[0];
    K(1, 2) = pp[1];
    return K;
}

cv::Vec4d Camera::getDistCoeff()
{
    return cv::Vec4d(intrinsic[1], intrinsic[2], 0.0, 0.0);
}

void Camera::updatePose(const cv::Matx33d &R, const cv::Vec3d &t)
{
    cv::Matx34d Rt = getExtrinsicMat();
    cv::Matx44d Rt_homo = cv::Matx44d::eye();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            Rt_homo(i, j) = Rt(i, j);

    cv::Matx34d T;
    cv::Matx44d T_homo = cv::Matx44d::eye();
    cv::hconcat(R, t, T);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            T_homo(i, j) = T(i, j);

    cv::Matx44d new_Rt_homo = Rt_homo * T_homo;
    cv::Matx33d new_R;
    cv::Vec3d new_t, new_rvec;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            new_R(i, j) = new_Rt_homo(i, j);
    for (int i = 0; i < 3; i++)
        new_t[i] = new_Rt_homo(i, 3);

    cv::Rodrigues(new_R, new_rvec);
    for (int i = 0; i < 3; i++) {
        extrinsic[i] = new_rvec[i];
        extrinsic[i+3] = new_t[i];
    }
}
