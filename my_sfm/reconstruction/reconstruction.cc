#include "reconstruction.h"
#include "images.h"
#include "keypoints.h"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/base.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "optimization.h"
#include "point_cloud.h"
#include "two_view_geometries.h"
#include <algorithm>
#include <cmath>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

/** select the next image which has the largest number of 2D-3D correspondences */
unsigned int Reconstruction::findNextBestView(std::vector<Match> &correspondences2d3d)
{
    unsigned int max_count = 0; // counting number of 3d points seen
    std::vector<Match> temp;
    unsigned int chosen_img_id = 0;
    std::unordered_map<Observation, unsigned int, ObservationHash> observation_to_point3D;

    for (unsigned int i = 0; i < pointcloud.points.size(); i++) {
        for (const Observation &observation : pointcloud.points[i].observations) {
            observation_to_point3D[observation] = i;
        }
    }

    std::vector<Match> matches;
    cv::Mat F;
    std::vector<bool> seen(pointcloud.points.size(), false);
    for (unsigned int i = 0; i < img_->getNumImgs(); i++) {
        if (registered_img_ids.count(i) > 0)
            continue; // dont care
        temp.clear();
        std::fill(seen.begin(), seen.end(), false);
        for (unsigned int j : registered_img_ids) {
            if (i < j) {
                if (!two_view_db_->Retrieve(i, j, matches, F)) continue;
                for (const Match &m : matches) {
                    const Observation o{j, m.idx2};
                    if (observation_to_point3D.count(o) > 0 && !seen[observation_to_point3D[o]]) {
                        temp.push_back(Match{m.idx1, observation_to_point3D[o]});
                        seen[observation_to_point3D[o]] = true;
                    }
                }
            } else {
                if(!two_view_db_->Retrieve(j, i, matches, F)) continue;
                for (const Match &m : matches) {
                    const Observation o{j, m.idx1};
                    if (observation_to_point3D.count(o) > 0 && !seen[observation_to_point3D[o]]) {
                        temp.push_back(Match{m.idx2, observation_to_point3D[o]});
                        seen[observation_to_point3D[o]] = true;
                    }
                }
            }
        }
        if (temp.size() > max_count) {
            chosen_img_id = i;
            max_count = temp.size();
            correspondences2d3d = temp;
        }
    }
    return chosen_img_id;
}

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
    std::vector<cv::Point2f> points1, points2,
    points1_filtered, points2_filtered,
    points1_undistorted, points2_undistorted;
    // choose initial 2 non-panoramic views and get E
    unsigned int img1_id, img2_id;
    std::vector<Match> inlier_correspondences, inlier_correspondences_filtered;
    cv::Mat F, E;

    two_view_db_->RetrieveBestTwoView(img1_id, img2_id, inlier_correspondences, F);

    cv::Matx33d K1 = cameras[img1_id].getIntrinsicMat();
    cv::Matx33d K2 = cameras[img2_id].getIntrinsicMat();
    cv::Vec4d distCoeffs1 = cameras[img1_id].getDistCoeff();
    cv::Vec4d distCoeffs2 = cameras[img2_id].getDistCoeff();
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
    cv::Mat points3D, points4D, img1, img2, inlier_mask;
    unsigned int N = inlier_correspondences.size();
    cv::recoverPose(
        points1, points2, K1, distCoeffs1, K2, distCoeffs2, E, R, t,
        cv::RANSAC, 0.999, 1.0, inlier_mask
    );
    unsigned int N_filtered = 0;
    for (unsigned int i = 0; i < N; i++) {
        if (inlier_mask.at<uchar>(i) != 0) {
            points1_filtered.push_back(points1[i]);
            points2_filtered.push_back(points2[i]);
            inlier_correspondences_filtered.push_back(inlier_correspondences[i]);
           N_filtered++;
        }
    }
    cameras[img1_id].updatePose(cv::Matx33d::eye(), cv::Vec3d::zeros());
    cameras[img2_id].updatePose(R, t); // this is world-to-cam matrix
    // undistort points already return pixels in normalized space so take Rt as P
    cv::undistortPoints(points1_filtered, points1_undistorted, K1, distCoeffs1);
    cv::undistortPoints(points2_filtered, points2_undistorted, K2, distCoeffs2);
    cv::triangulatePoints(
        cameras[img1_id].getExtrinsicMat(), cameras[img2_id].getExtrinsicMat(), points1_undistorted, points2_undistorted, points4D
    );
    cv::convertPointsFromHomogeneous(points4D.t(), points3D);
    // add them to the point cloud
    img1 = img_->loadImages(img1_id);
    img2 = img_->loadImages(img2_id);
    for (unsigned int i = 0; i < N_filtered; i++) {
        cv::Vec3b color1 = img1.at<cv::Vec3b>((int) std::round(points1_filtered[i].y), (int) std::round(points1_filtered[i].x));
        cv::Vec3b color2 = img2.at<cv::Vec3b>((int) std::round(points2_filtered[i].y), (int) std::round(points2_filtered[i].x));
        cv::Vec3b color = (color1 + color2) / 2;
        cv::Vec3f xyz = points3D.at<cv::Vec3f>(i, 0);
        cv::Vec3d p(xyz[0], xyz[1], xyz[2]);
        Point point(p, color);
        point.addObservation(img1_id, inlier_correspondences_filtered[i].idx1);
        point.addObservation(img2_id, inlier_correspondences_filtered[i].idx2);
        pointcloud.addPoint(point);
    }
    // lets go bundle adjustment
    optimize(pointcloud, cameras, key_points_db_, img1_id, img2_id, true);
    std::cout << cameras[img1_id].getIntrinsicMat() << '\n' << cameras[img2_id].getIntrinsicMat() << '\n' << cameras[img2_id].getExtrinsicMat() << '\n';
    std::cout << cameras[img1_id].getDistCoeff() << '\n' << cameras[img2_id].getDistCoeff() << '\n';
    registered_img_ids.insert(img1_id);
    registered_img_ids.insert(img2_id);
    for (const Point &p : pointcloud.points) {
        std::cout << p.pt << '\n';
    }
    return true;
}

bool Reconstruction::ImageRegistration()
{
    std::vector<Match> correspondences2d3d;
    unsigned int next_img_id = findNextBestView(correspondences2d3d);
    std::vector<cv::Point2d> img_points;
    std::vector<cv::Point3d> obj_points;
    std::vector<cv::KeyPoint> keypoints;
    cv::Vec3d rvec, tvec;
    cv::Matx33d R;
    key_points_db_->Retrieve(next_img_id, keypoints);
    // add the observation
    for (const Match &correspondence : correspondences2d3d) {
        pointcloud.points[correspondence.idx2].addObservation(next_img_id, correspondence.idx1);
        img_points.push_back(static_cast<cv::Point2d>(keypoints[correspondence.idx1].pt));
        obj_points.emplace_back(pointcloud.points[correspondence.idx2].pt);
    }
    CV_Assert(obj_points.size() == img_points.size());
    CV_Assert(obj_points.size() == correspondences2d3d.size());
    // solving pnp problems
    cv::solvePnPRansac(
        obj_points,
        img_points,
        cameras[next_img_id].getIntrinsicMat(),
        cameras[next_img_id].getDistCoeff(),
        rvec, tvec,
        false, 100, 10.0, 0.99, cv::noArray(),
        cv::SOLVEPNP_EPNP);
    cv::Rodrigues(rvec, R);
    cameras[next_img_id].updatePose(R, tvec);
    // triangulate new 3d points with previously registered views
    std::cout << next_img_id << " " << correspondences2d3d.size() << "\n" << cameras[next_img_id].getExtrinsicMat() << '\n';
    return true;
}
