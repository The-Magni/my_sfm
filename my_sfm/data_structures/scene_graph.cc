#include "scene_graph.h"

SceneGraph::SceneGraph(unsigned int num_imgs)
{
    adjList = std::vector<std::vector<AdjListElement>> (num_imgs);
}

void SceneGraph::addEdge(
    const Correspondences &inlier_correspondences,
    const cv::Mat &F,
    unsigned int img1_id,
    unsigned int img2_id
) {
    EdgeData e{inlier_correspondences, F};
    if (img1_id < img2_id) {
        AdjListElement element{img2_id, e};
        adjList.at(img1_id).push_back(element);
    } else if (img2_id < img1_id) {
        AdjListElement element{img1_id, e};
        adjList.at(img2_id).push_back(element);
    }
}

bool SceneGraph::getEdge(unsigned int img1_id, unsigned int img2_id, EdgeData &e)
{
    if (img1_id >= img2_id) {
        std::cout << "Invalid order of images, first id must be smaller than second id";
        return false;
    }

    for (const AdjListElement &element : adjList[img1_id]) {
        if (element.image_id == img2_id) {
            e = element.edge;
            return true;
        }
    }
    return false; // edge doesnt exist
}
