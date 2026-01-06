#include "outlier_filtering.h"
#include "camera.h"
#include "keypoints.h"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "point_cloud.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

OutlierFiltering::OutlierFiltering(
    std::shared_ptr<KeyPointsDB> keypoints_db,
    double max_filter_reproj_error,
    double min_filter_tri_angle
) {
    keypoints_db_ = keypoints_db;
    max_filter_reproj_error_ = max_filter_reproj_error;
    min_filter_tri_angle_ = min_filter_tri_angle;
}

bool OutlierFiltering::isLargeReprojectionError(const cv::Vec3d &point3D, const cv::Point2f &point2D, const Camera &camera)
{
    cv::Vec2d predicted_vec2D = camera.project(point3D);
    cv::Point2f predicted_point2D(static_cast<float>(predicted_vec2D[0]), static_cast<float>(predicted_vec2D[1]));
    return cv::norm(point2D - predicted_point2D) >= max_filter_reproj_error_;
}


void OutlierFiltering::FilterObservationByReprojectionError(
    PointCloud &pointcloud,
    const std::vector<Camera> &cameras
) {
    std::unordered_map<unsigned int, std::vector<cv::KeyPoint>> cache;
    for (auto i = pointcloud.points.begin(); i != pointcloud.points.end();) {
        Point &p = *i;
        const Point copy = *i;
        for (auto j = p.observations.begin(); j != p.observations.end();) {
            // calculate reprojection error
            const Observation &o = *j;
            if (!cache.count(o.img_id)) keypoints_db_->Retrieve(o.img_id, cache[o.img_id]);
            if (isLargeReprojectionError(p.pt, cache[o.img_id].at(o.point_id).pt, cameras[o.img_id]))
                j = p.observations.erase(j);
            else
                j++;
        }
        // remove the point if it has 0 or 1 observation (underconstraint)
        if (p.observations.size() <= 1)
            i = pointcloud.points.erase(i);
        else
            i++;
    }
}

bool OutlierFiltering::isSmallTriAngle(const cv::Vec3d &point, const Camera &cam1, const Camera &cam2)
{
    double cos_theta, norm1, norm2;
    cv::Matx33d R1, R2;
    cv::Vec3d t1, t2, v1, v2;
    cam1.getPose(R1, t1);
    cam2.getPose(R2, t2);
    v1 = point - t1;
    v2 = point - t2;
    norm1 = cv::norm(v1);
    norm2 = cv::norm(v2);
    if (norm1 < 1e-9 || norm2 < 1e-9)
        return true;
    else
        cos_theta = std::clamp(v1.dot(v2) / (norm1 * norm2), -1.0, 1.0);

    return std::acos(cos_theta) * 180.0 / M_PI <= min_filter_tri_angle_;
}

void OutlierFiltering::FilterPointsByTriAngle(
    PointCloud &pointcloud,
    const std::vector<Camera> &cameras
) {
    bool is_filtered = false;
    for (auto it = pointcloud.points.begin(); it != pointcloud.points.end();) {
        is_filtered = false;
        const Point &p = *it;
        for (unsigned int i = 0; i < p.observations.size(); i++) {
            for (unsigned int j = i+1; j < p.observations.size(); j++) {
                const Camera &cam1 = cameras[p.observations[i].img_id];
                const Camera &cam2 = cameras[p.observations[j].img_id];
                if (isSmallTriAngle(p.pt, cam1, cam2)) {
                    is_filtered = true;
                    break;
                }
            }

            if (is_filtered)
                break;
        }
        if (is_filtered)
            it = pointcloud.points.erase(it);
        else
            it++;
    }
}

bool OutlierFiltering::isBehindCam(const cv::Vec3d &point, const Camera &cam)
{
    cv::Matx34d Rt = cam.getExtrinsicMat();
    cv::Matx33d R(
        Rt(0, 0), Rt(0, 1), Rt(0, 2),
        Rt(1, 0), Rt(1, 1), Rt(1, 2),
        Rt(2, 0), Rt(2, 1), Rt(2, 2)
    );
    cv::Vec3d t(Rt(0, 3), Rt(1, 3), Rt(2, 3));
    cv::Vec3d point3D_cam = R * point + t;
    return point3D_cam[2] <= 0;
}

void OutlierFiltering::FilteringPointsBehindCam(
    PointCloud &pointcloud,
    const std::vector<Camera> &cameras
) {
    for (auto i = pointcloud.points.begin(); i != pointcloud.points.end(); ) {
        Point &p = *i;
        for (auto j = p.observations.begin(); j != p.observations.end(); ) {
            const Observation &o = *j;
            if (isBehindCam(p.pt, cameras[o.img_id]))
                j = p.observations.erase(j);
            else
                j++;
        }
        if (p.observations.size() <= 1)
            i = pointcloud.points.erase(i);
        else
            i++;
    }
}


void OutlierFiltering::Process(PointCloud &pointcloud, const std::vector<Camera> &cameras)
{
    FilterObservationByReprojectionError(pointcloud, cameras);
    FilteringPointsBehindCam(pointcloud, cameras);
    FilterPointsByTriAngle(pointcloud, cameras);
}
