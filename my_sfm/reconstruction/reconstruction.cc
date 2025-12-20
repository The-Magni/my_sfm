#include "reconstruction.h"
#include "images.h"
#include "keypoints.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/types.hpp"
#include "two_view_geometries.h"
#include <memory>

cv::Mat Reconstruction::Rt2T(const cv::Mat &R, const cv::Mat &t)
{
    cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
    R.copyTo(T(cv::Rect(0, 0, 3, 3)));
    t.copyTo(T(cv::Rect(3, 0, 1, 3)));
    return T;
}

/*
 * Assume the following
 * All images are captured by the same camera (same instrinic and same resolution)
 * Camera are uncalirated
 * Using pinhole model
 */
bool Reconstruction::Init(std::shared_ptr<TwoViewGeometriesDB> two_view_db, std::shared_ptr<Images> img, std::shared_ptr<KeyPointsDB> key_points_db)
{
    two_view_db_ = two_view_db;
    img_ = img;
    key_points_db_ = key_points_db;
    // init K to default value due to uncalirated cam
    cv::Mat R, t, P1, P2 = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat sample = img_->loadImages(0);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::Point2f> points1, points2;
    double f = 1.2 * std::max(sample.rows, sample.cols);
    double px = (double) sample.cols / 2;
    double py = (double) sample.rows / 2;
    K.at<double>(0, 0) = f;
    K.at<double>(0, 2) = px;
    K.at<double>(1, 1) = f;
    K.at<double>(1, 2) = py;
    // choose initial 2 non-panoramic views and get E
    unsigned int img1_id, img2_id;
    std::vector<Match> inlier_correspondences;
    cv::Mat F, E;
    two_view_db_->RetrieveBestTwoView(img1_id, img2_id, inlier_correspondences, F);
    E = K.t() * F * K;
    // retrieve keypoints and construct matching points
    key_points_db_->Retrieve(img1_id, keypoints1);
    key_points_db_->Retrieve(img2_id, keypoints2);
    for (const Match &m : inlier_correspondences) {
        points1.push_back(keypoints1.at(m.idx1).pt);
        points2.push_back(keypoints2.at(m.idx2).pt);
    }

    // lets go triangulation
    cv::recoverPose(E, points1, points2, R, t, f, cv::Point2d(px, py), cv::noArray());
    P1 = K * cv::Mat::eye(4, 4, CV_64F);
    P2 = K * Rt2T(R, t);
    cv::triangulatePoints(P1, P2, points1, points2, points4D);

    // lets go bundle adjustment

    return true;
}
