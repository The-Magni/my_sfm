#pragma once

#include "images.h"
#include "keypoints.h"
#include "two_view_geometries.h"
#include <memory>

class Reconstruction {
    private:
        std::shared_ptr<TwoViewGeometriesDB> two_view_db_{nullptr};
        std::shared_ptr<Images> img_{nullptr};
        std::shared_ptr<KeyPointsDB> key_points_db_{nullptr};
        cv::Mat points4D;

        cv::Mat Rt2T(const cv::Mat &R, const cv::Mat &t);

    public:
        bool Init(std::shared_ptr<TwoViewGeometriesDB> two_view_db, std::shared_ptr<Images> img, std::shared_ptr<KeyPointsDB> key_points_db);
};
