#include "outlier_filtering.h"
#include "keypoints.h"
#include "opencv2/core/matx.hpp"
#include "opencv2/core/types.hpp"
#include "point_cloud.h"
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

OutlierFiltering::OutlierFiltering(
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int max_filter_reproj_error,
    unsigned int min_filter_tri_angle
) {
    keypoints_db_ = keypoints_db;
    max_filter_reproj_error_ = max_filter_reproj_error;
    min_filter_tri_angle_ = min_filter_tri_angle;
}

void OutlierFiltering::FilterObservationByReprojectionError(PointCloud &pointcloud, const std::vector<Camera> &cameras)
{
    std::unordered_map<unsigned int, std::vector<cv::KeyPoint>> cache;
    for (auto i = pointcloud.points.begin(); i != pointcloud.points.end();) {
        Point &p = *i;
        for (auto j = p.observations.begin(); j != p.observations.end();) {
            // calculate reprojection error
            const Observation &o = *j;
            if (!cache.count(o.img_id))
                keypoints_db_->Retrieve(o.img_id, cache[o.img_id]);
            cv::Point2f observed_point2D = cache[o.img_id].at(o.point_id).pt;
            cv::Vec2d predicted_vec2D = cameras[o.img_id].project(p.pt);
            cv::Point2f predicted_point2D(static_cast<float>(predicted_vec2D[0]), static_cast<float>(predicted_vec2D[1]));
            // remove points with larg reprojection error
            if (cv::norm(observed_point2D - predicted_point2D) >= max_filter_reproj_error_)
                j = p.observations.erase(j);
            else
                j++;
        }
        // remove the point if it has 0 observation
        if (p.observations.size() == 0)
            i = pointcloud.points.erase(i);
        else
            i++;
    }
}

void OutlierFiltering::FilterPointsByTriAngle(PointCloud &pointcloud, const std::vector<Camera> &cameras)
{
    bool is_filtered = false;
    for (auto it = pointcloud.points.begin(); it != pointcloud.points.end();) {
        is_filtered = false;
        const Point &p = *it;
        for (unsigned int i = 0; i < p.observations.size(); i++) {
            for (unsigned int j = i+1; j < p.observations.size(); j++) {
                const Camera &cam1 = cameras[p.observations[i].img_id];
                const Camera &cam2 = cameras[p.observations[j].img_id];
                double cos_theta, norm1, norm2;
                cv::Matx33d R1, R2;
                cv::Vec3d t1, t2, v1, v2;
                cam1.getPose(R1, t1);
                cam2.getPose(R2, t2);
                v1 = p.pt - t1;
                v2 = p.pt - t2;
                norm1 = cv::norm(v1);
                norm2 = cv::norm(v2);
                if (norm1 < 1e-9 || norm2 < 1e-9)
                    cos_theta = 1; // mean that theta = 0, will be filtered
                else
                    cos_theta = v1.dot(v2) / (norm1 * norm2);
                if (std::acos(cos_theta) * 180.0 / M_PI <= min_filter_tri_angle_) {
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

void OutlierFiltering::Process(PointCloud &pointcloud, const std::vector<Camera> &cameras)
{
    FilterObservationByReprojectionError(pointcloud, cameras);
    FilterPointsByTriAngle(pointcloud, cameras);
}
