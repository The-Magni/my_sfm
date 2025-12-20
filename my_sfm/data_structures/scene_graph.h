#pragma once

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

struct Correspondences {
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
};

struct EdgeData {
    Correspondences inlier_correspondences;
    cv::Mat F;
};

struct AdjListElement {
    unsigned int image_id;
    EdgeData edge;
};

class SceneGraph {
    private:
        std::vector<std::vector<AdjListElement>> adjList;

    public:
        SceneGraph(unsigned int num_imgs);

        void addEdge(
            const Correspondences &inlier_correspondences,
            const cv::Mat &F,
            unsigned int img1_id,
            unsigned int img2_id
        );

        bool getEdge(unsigned int img1_id, unsigned int img2_id, EdgeData &e);
};
