#pragma once

#include "camera.h"
#include "keypoints.h"
#include "point_cloud.h"
#include <memory>

void optimize(
    PointCloud &pointcloud,
    std::vector<Camera> &cameras,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int first_img_id,
    unsigned int second_img_id,
    bool is_first_two_views
);
