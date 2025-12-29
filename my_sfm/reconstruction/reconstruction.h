#pragma once

#include "camera.h"
#include "images.h"
#include "keypoints.h"
#include "two_view_geometries.h"
#include "point_cloud.h"
#include <memory>
#include <vector>

class Reconstruction {
    private:
        std::shared_ptr<TwoViewGeometriesDB> two_view_db_{nullptr};
        std::shared_ptr<Images> img_{nullptr};
        std::shared_ptr<KeyPointsDB> key_points_db_{nullptr};
        std::vector<Camera> cameras;
        PointCloud pointcloud;
        std::set<unsigned int> registered_img_ids;

        unsigned int findNextBestView(std::vector<Match> &correspondences2d3d);

    public:
        bool Init(std::shared_ptr<TwoViewGeometriesDB> two_view_db, std::shared_ptr<Images> img, std::shared_ptr<KeyPointsDB> key_points_db);
        bool ImageRegistration();
};
