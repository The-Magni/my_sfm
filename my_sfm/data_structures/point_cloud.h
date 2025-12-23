#pragma once

#include "opencv2/core/matx.hpp"
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

    void addPoint(const Point &point);
};
