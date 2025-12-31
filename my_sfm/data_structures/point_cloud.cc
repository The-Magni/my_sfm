#include "point_cloud.h"

void PointCloud::addPoint(const Point &point)
{
    points.push_back(point);
    for (const Observation &o : point.observations) {
        observation_to_point3d[o] = points.size() - 1;
    }
}

void PointCloud::addObservation(unsigned int idx, unsigned int img_id, unsigned int point_id)
{
    points.at(idx).addObservation(img_id, point_id);
    const Observation o(img_id, point_id);
    observation_to_point3d[o] = idx;
}
