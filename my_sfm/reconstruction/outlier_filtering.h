#pragma once

#include "camera.h"
#include "keypoints.h"
#include "point_cloud.h"
#include <memory>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class OutlierFiltering {
    private:
        double max_filter_reproj_error_;
        double min_filter_tri_angle_; // in degrees
        std::shared_ptr<KeyPointsDB> keypoints_db_{nullptr};

        void FilterObservationByReprojectionError(
            PointCloud &pointcloud,
            const std::vector<Camera> &cameras
        );

        void FilterPointsByTriAngle(
            PointCloud &pointcloud,
            const std::vector<Camera> &cameras
        );

        void FilteringPointsBehindCam(
            PointCloud &pointcloud,
            const std::vector<Camera> &cameras
        );

    public:
        OutlierFiltering(
            std::shared_ptr<KeyPointsDB> keypoints_db,
            double max_filter_reproj_error,
            double min_filter_tri_angle
        );

        bool isLargeReprojectionError(const cv::Vec3d &point3D, const cv::Point2f &point2D, const Camera &camera);

        bool isSmallTriAngle(const cv::Vec3d &point, const Camera &cam1, const Camera &cam2);

        bool isBehindCam(const cv::Vec3d &point, const Camera &cam);

        void Process(
            PointCloud &pointcloud,
            const std::vector<Camera> &cameras
        );
};
