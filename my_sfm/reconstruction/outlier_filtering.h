#pragma once

#include "camera.h"
#include "keypoints.h"
#include "point_cloud.h"
#include <memory>
#include <vector>

class OutlierFiltering {
    private:
        double max_filter_reproj_error_;
        double min_filter_tri_angle_; // in degrees
        std::shared_ptr<KeyPointsDB> keypoints_db_{nullptr};

        void FilterObservationByReprojectionError(PointCloud &pointcloud, const std::vector<Camera> &cameras);

        void FilterPointsByTriAngle(PointCloud &pointcloud, const std::vector<Camera> &cameras);

        void FilteringPointsBehindCam(PointCloud &pointcloud, const std::vector<Camera> &cameras);

    public:
        OutlierFiltering(
            std::shared_ptr<KeyPointsDB> keypoints_db,
            double max_filter_reproj_error,
            double min_filter_tri_error_
        );

        void Process(PointCloud &pointcloud, const std::vector<Camera> &cameras);
};
