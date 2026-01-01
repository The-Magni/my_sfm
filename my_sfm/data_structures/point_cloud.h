#pragma once

#include "opencv2/core/matx.hpp"
#include <cstddef>
#include <functional>
#include <pstl/glue_algorithm_defs.h>
#include <unordered_map>
#include <vector>

/** storing 3D-2D correspondences */
struct Observation {
    unsigned int img_id;
    unsigned int point_id; /* index in the two_view_geometries database, index to the keypoints */

    Observation(unsigned int img_id, unsigned int point_id)
    {
        this->img_id = img_id;
        this->point_id = point_id;
    }

    bool operator==(const Observation &other) const
    {
        return img_id == other.img_id && point_id == other.point_id;
    }
};

struct ObservationHash {
    std::size_t operator()(const Observation &o) const noexcept
    {
        return std::hash<unsigned int>()(o.img_id) ^ (std::hash<unsigned int>()(o.point_id) << 1);
    }
};

struct Point {
    cv::Vec3d pt;
    cv::Vec3b color;
    std::vector<Observation> observations;

    Point(const cv::Vec3d &pt, const cv::Vec3b &color)
    {
        this->pt = pt;
        this->color = color;
    }

    void addObservation(unsigned int img_id, unsigned int point_id)
    {
        observations.push_back(Observation(img_id, point_id));
    }

    double *getPt()
    {
        return pt.val;
    }
};

struct PointCloud {
    std::vector<Point> points;
    std::unordered_map<Observation, unsigned int, ObservationHash> observation_to_point3d;

    void addPoint(const Point &point);

    void addObservation(unsigned int idx, unsigned int img_id, unsigned int point_id);

    void rebuildMap();
};
