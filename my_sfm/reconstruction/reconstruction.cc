#include "reconstruction.h"
#include "images.h"
#include "keypoints.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "optimization.h"
#include "two_view_geometries.h"
#include <cmath>
#include <memory>
#include <vector>

bool Reconstruction::Init(std::shared_ptr<TwoViewGeometriesDB> two_view_db, std::shared_ptr<Images> img, std::shared_ptr<KeyPointsDB> key_points_db)
{
    two_view_db_ = two_view_db;
    img_ = img;
    key_points_db_ = key_points_db;
    // initialize camera parameters to default value
    cameras.reserve(img_->getNumImgs());
    cv::Mat sample;
    for (unsigned int i = 0; i < img_->getNumImgs(); i++) {
        sample = img_->loadImages(i);
        cameras.emplace_back(sample.cols, sample.rows); // initialuze the camera parameters to default value
    }
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::Point2f> points1, points2, points1_undistorted, points2_undistorted;
    // choose initial 2 non-panoramic views and get E
    unsigned int img1_id, img2_id;
    std::vector<Match> inlier_correspondences;
    cv::Mat F, E;

    two_view_db_->RetrieveBestTwoView(img1_id, img2_id, inlier_correspondences, F);

    // do something fishy
    cv::Matx33d K1 = cameras[img1_id].getIntrinsicMat();
    cv::Matx33d K2 = cameras[img2_id].getIntrinsicMat();
    cv::Vec4d distCoeffs1 = cameras[img1_id].getDistCoeff();
    cv::Vec4d distCoeffs2 = cameras[img2_id].getDistCoeff();
    two_view_db_->RetrieveBestTwoView(img1_id, img2_id, inlier_correspondences, F);
    E = K2.t() * F * K1;
    // retrieve keypoints and construct matching points
    key_points_db_->Retrieve(img1_id, keypoints1);
    key_points_db_->Retrieve(img2_id, keypoints2);
    for (const Match &m : inlier_correspondences) {
        points1.push_back(keypoints1.at(m.idx1).pt);
        points2.push_back(keypoints2.at(m.idx2).pt);
    }

    // lets go triangulation
    cv::Matx33d R;
    cv::Vec3d t;
    cv::Mat points3D, points4D, img1, img2;
    unsigned int N = inlier_correspondences.size();
    cv::undistortPoints(points1, points1_undistorted, K1, distCoeffs1);
    cv::undistortPoints(points2, points2_undistorted, K2, distCoeffs2);
    // already normalize t to 1
    cv::recoverPose(E, points1_undistorted, points2_undistorted, R, t);
    cameras[img2_id].updatePose(R, t);
    // undistort points already return pixels in normalized space so take Rt as P
    cv::triangulatePoints(cameras[img1_id].getExtrinsicMat(), cameras[img2_id].getExtrinsicMat(), points1_undistorted, points2_undistorted, points4D);
    cv::convertPointsFromHomogeneous(points4D.t(), points3D);
    // add them to the point cloud
    img1 = img_->loadImages(img1_id);
    img2 = img_->loadImages(img2_id);
    for (unsigned int i = 0; i < N; i++) {
        cv::Vec3b color1 = img1.at<cv::Vec3b>((int) std::round(points1[i].y), (int) std::round(points1[i].x));
        cv::Vec3b color2 = img2.at<cv::Vec3b>((int) std::round(points2[i].y), (int) std::round(points2[i].x));
        cv::Vec3b color = (color1 + color2) / 2;
        cv::Vec3d xyz(
            points3D.at<double>(0, i),
            points3D.at<double>(1, i),
            points3D.at<double>(2, i)
        );
        Point point(xyz, color);
        point.addObservation(img1_id, inlier_correspondences[i].idx1);
        point.addObservation(img2_id, inlier_correspondences[i].idx2);
        pointcloud.addPoint(point);
    }
    // lets go bundle adjustment
    optimize(pointcloud, cameras, key_points_db_, img1_id, img2_id);

    return true;
}
