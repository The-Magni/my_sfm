#pragma once

#include "camera.h"
#include "keypoints.h"
#include "point_cloud.h"
#include <memory>
#include <vector>

class OutlierFiltering {
    private:
        unsigned int max_filter_reproj_error_;
        unsigned int min_filter_tri_angle_; // in degrees
        std::shared_ptr<KeyPointsDB> keypoints_db_{nullptr};

        OutlierFiltering(
            std::shared_ptr<KeyPointsDB> keypoints_db,
            unsigned int max_filter_reproj_error,
            unsigned int min_filter_tri_error_
        );

        void FilterObservationByReprojectionError(PointCloud &pointcloud, const std::vector<Camera> &cameras);

        void FilterPointsByTriAngle(PointCloud &pointcloud, const std::vector<Camera> &camera);

    public:
        void Process(PointCloud &pointcloud, const std::vector<Camera> &cameras);
};
