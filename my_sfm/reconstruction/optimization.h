#pragma once

#include "camera.h"
#include "keypoints.h"
#include "point_cloud.h"
#include <memory>

void optimize(
    PointCloud &pointcloud,
    std::vector<Camera> &cameras,
    std::shared_ptr<KeyPointsDB> keypoints_db,
    unsigned int fixed_img_id
);
