#pragma once

#include "camera.h"
#include "images.h"
#include "keypoints.h"
#include "outlier_filtering.h"
#include "two_view_geometries.h"
#include "point_cloud.h"
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

class Reconstruction {
    private:
        std::shared_ptr<TwoViewGeometriesDB> two_view_db_{nullptr};
        std::shared_ptr<Images> img_{nullptr};
        std::shared_ptr<KeyPointsDB> key_points_db_{nullptr};
        std::shared_ptr<OutlierFiltering> outlier_filtering_{nullptr};
        std::vector<Camera> cameras;
        PointCloud pointcloud;
        std::set<unsigned int> registered_img_ids;
        std::array<unsigned int, 2> first_two_views;

        unsigned int findNextBestView(std::vector<Match> &correspondences2d3d);

    public:
        bool Init(std::shared_ptr<TwoViewGeometriesDB> two_view_db, std::shared_ptr<Images> img, std::shared_ptr<KeyPointsDB> key_points_db);

        bool ImageRegistration();

        bool IncrementalReconstruction();

        bool Write(const std::string &filepath);

        void ReTriangulation();

        std::vector<Point> Triangulation(
            unsigned int img1_id,
            unsigned int img2_id,
            const std::vector<cv::Point2f> &points1,
            const std::vector<cv::Point2f> &points2,
            const cv::Mat &img1,
            const cv::Mat &img2
        );

        void TrackExtension();
};
